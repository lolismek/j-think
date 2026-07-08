"""
Latent-workspace thinking engine (single GPU, standard full-attention model).
Default target: Qwen3-4B (tied embeddings -> W_a ~ I, clean re-embed).

Loop (one position at a time after prefill; each single-query step attends to
ALL of its layer's cached keys, so per-layer cache lengths may differ -> latent
positions live only in workspace layers):

  prefill -> DynamicCache (all layers, len P), h_j@lastpos, h_final@lastpos
  latent step x K:  hatL = mu_fin + J_mot (h_j - mu_mot);  e = fnorm(hatL) @ W_a;
                    h_in = mu_ent + J_in (e - mu_e);  h_j = blocks[i..j](h_in)
  answer: feed cue tokens (full stack) then greedy-decode (full stack).

Env: MODEL, I_ENTRY, J_EXIT, MOTOR_AFFINE. Import this module to load once.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
TAG = os.environ.get("TAG", "qwen3-4b")
I_ENTRY = int(os.environ.get("I_ENTRY", 7))
J_EXIT = int(os.environ.get("J_EXIT", 28))
MOTOR_AFFINE = os.environ.get("MOTOR_AFFINE", "1") == "1"
import transformers, jlens
from transformers import DynamicCache
DEV = "cuda:0"

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
hf.eval()
model = jlens.from_hf(hf, tok)
text = model._text_module; layers = model.layers; rotary = text.rotary_emb
embed = model._embed_tokens; fnorm = model._final_norm
NL = model.n_layers; TGT_SENS = I_ENTRY - 1; FINAL = NL - 1
print(f"loaded {MODEL}: L={NL} d={model.d_model} workspace=[{I_ENTRY},{J_EXIT}]", flush=True)

# ---------------- bridges ----------------
Wa = torch.load(f"{OUT}/{TAG}_W_a.pt")["W_a"].to(DEV)
Jin = torch.load(f"{OUT}/{TAG}_Jin.pt")["Jin"].to(DEV)
mu = {k: v.float().to(DEV) for k, v in torch.load(f"{OUT}/{TAG}_stage0_stats.pt")["mu"].items()}
lens = jlens.JacobianLens.from_pretrained(f"{OUT}/{TAG}_lens.pt") if os.path.exists(f"{OUT}/{TAG}_lens.pt") \
    else jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", revision="main")
Jmot = lens.jacobians[J_EXIT].float().to(DEV)

def bridge(h_j, target_norm=None):  # h_j [d] -> h_in [1,1,d]
    h = h_j.float()
    hatL = (mu["h_fin"] + (h - mu["h_mot"]) @ Jmot.T) if MOTOR_AFFINE else (h @ Jmot.T)
    e = fnorm(hatL.to(torch.bfloat16)).float() @ Wa
    h_in = mu["h_ent"] + (e - mu["e"]) @ Jin.T
    if target_norm is not None:        # keep on the workspace manifold (J^in has large gain)
        h_in = h_in * (target_norm / h_in.norm().clamp_min(1e-6))
    return h_in.to(torch.bfloat16).view(1, 1, -1)

# ---------------- manual single-position runner ----------------
def cossin(pos):
    pid = torch.tensor([[pos]], device=DEV)
    cos, sin = rotary(torch.zeros(1, 1, 1, device=DEV, dtype=torch.bfloat16), pid)
    return (cos, sin), pid

def run_through(hidden, pos, layer_ids, cache):
    pe, pid = cossin(pos)
    for l in layer_ids:
        out = layers[l](hidden, attention_mask=None, position_ids=pid,
                        past_key_values=cache, use_cache=True, position_embeddings=pe)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden

@torch.no_grad()
def prefill(prompt_ids):
    cache = DynamicCache(); cap = {}
    grab = lambda name, idx: layers[idx].register_forward_hook(
        lambda m, i, o: cap.__setitem__(name, (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))
    hs = [grab("hj", J_EXIT), grab("hfin", FINAL), grab("hent", TGT_SENS)]
    text(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    for h in hs: h.remove()
    return cache, cap["hj"], cap["hfin"], cap["hent"], prompt_ids.shape[1]

@torch.no_grad()
def latent_generate(prompt_ids, K, cue_ids, max_new=24, stop_ids=None):
    stop = set(stop_ids or []); stop.add(tok.eos_token_id)
    cache, hj, hfin, hent, P = prefill(prompt_ids); pos = P
    tgt = hent.float().norm()
    for _ in range(K):
        h_in = bridge(hj, tgt)
        hj = run_through(h_in, pos, list(range(I_ENTRY, J_EXIT + 1)), cache)[0, -1]; pos += 1
    logits = model.unembed(hfin.view(1, 1, -1))[0, -1]
    for c in cue_ids:
        hid = embed(torch.tensor([[c]], device=DEV)).to(torch.bfloat16)
        hid = run_through(hid, pos, list(range(NL)), cache); pos += 1
        logits = model.unembed(hid)[0, -1]
    out = []
    for _ in range(max_new):
        nxt = logits.argmax().item()
        if nxt in stop: break
        out.append(nxt)
        hid = embed(torch.tensor([[nxt]], device=DEV)).to(torch.bfloat16)
        hid = run_through(hid, pos, list(range(NL)), cache); pos += 1
        logits = model.unembed(hid)[0, -1]
    return out

# ---------------- self-test: manual == generate ----------------
@torch.no_grad()
def manual_decode(prompt_ids, n_new):
    cache, hj, hfin, hent, P = prefill(prompt_ids)
    nxt = model.unembed(hfin.view(1, 1, -1))[0, -1].argmax().item()
    out = [nxt]; pos = P
    for _ in range(n_new - 1):
        hid = embed(torch.tensor([[nxt]], device=DEV)).to(torch.bfloat16)
        hid = run_through(hid, pos, list(range(NL)), cache)
        nxt = model.unembed(hid)[0, -1].argmax().item(); out.append(nxt); pos += 1
    return out

@torch.no_grad()
def selftest():
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(DEV)
    gen = hf.generate(ids, max_new_tokens=6, do_sample=False)[0, ids.shape[1]:].tolist()
    man = manual_decode(ids, 6)
    print("generate:", gen, "\nmanual  :", man, "\nSELF-TEST", "PASS" if gen == man else "FAIL", flush=True)
    return gen == man

if __name__ == "__main__":
    if not selftest():
        raise SystemExit("runtime mismatch")
    ids = tok("A robot must think carefully before answering the question.",
              return_tensors="pt").input_ids.to(DEV)
    cache, hj, hfin, hent, P = prefill(ids); pos = P
    tgt = hent.float().norm()
    import torch.nn.functional as Fn
    print(f"\n[LATENT TRAJECTORY] P={P} |h_ent|={tgt:.1f} |h_j|0={hj.float().norm():.1f}")
    prev = None
    for k in range(8):
        h_in = bridge(hj, tgt)
        c = Fn.cosine_similarity(h_in.view(-1).float(), prev.view(-1).float(), dim=0).item() if prev is not None else float("nan")
        prev = h_in
        hj = run_through(h_in, pos, list(range(I_ENTRY, J_EXIT + 1)), cache)[0, -1]; pos += 1
        print(f"  step {k+1}: |h_in|={h_in.float().norm():.1f} |h_j|={hj.float().norm():.1f} cos(in,prev)={c:.3f}")
    print("Stage-1 mechanics OK.", flush=True)

"""
Does passing e through the REAL lower layers (0..i-1) work, where J_in fails?
Same input e = fnorm(h_L[p]), same target = real h_ent[p]; only the transform differs.

Probe: replace position p's layer-0 input embedding with e[p] (all other positions real),
run real layers 0..TGT_SENS, read position p's output = h_ent_from_e[p]. One modified
position per batch row (so causal context for that position stays real).

Variants of the injected vector:
  raw        : e[p]                       (norm ~137)
  scaled     : e[p] rescaled to |emb[p]|  (embedding scale; residual stream in-distribution)
  sanity     : emb[p] unmodified          (must reproduce real h_ent -> cos 1.0, validates harness)
Baseline: J_in(e[p]) vs same target (the current pipeline's sensory hop).
Metric: uncentered cos AND centered signal cos(.-mu_ent, h_ent-mu_ent).
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka"); MODEL = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"
I_ENTRY = int(os.environ.get("I_ENTRY", 7)); SKIP = 16; DEV = "cuda:0"
N_PROMPT = int(os.environ.get("N_PROMPT", 30)); MAX_LEN = 64
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); text = model._text_module; layers = model.layers
fnorm = model._final_norm; embed = model._embed_tokens
NL, D = model.n_layers, model.d_model; TGT_SENS, FIN = I_ENTRY - 1, NL - 1
Jin = torch.load(f"{OUT}/{TAG}_Jin.pt")["Jin"].float().to(DEV)
mu = {k: v.float().to(DEV) for k, v in torch.load(f"{OUT}/{TAG}_stage0_stats.pt")["mu"].items()}
print(f"{MODEL} workspace entry i={I_ENTRY} (read layer {TGT_SENS} output)", flush=True)

cap = {}
def mk(n):
    def h(m, i, o): cap[n] = (o[0] if isinstance(o, tuple) else o)
    return h
def real_fwd(ids):  # full stack, capture h_ent (layer TGT_SENS out) and h_L (final out)
    cap.clear()
    hs = [layers[TGT_SENS].register_forward_hook(mk("ent")), layers[FIN].register_forward_hook(mk("fin"))]
    with torch.no_grad(): text(input_ids=ids, use_cache=False)
    for h in hs: h.remove()
    return cap["ent"][0].float(), cap["fin"][0].float()

full = text.layers
def lower_fwd(batch):  # run only layers 0..TGT_SENS on inputs_embeds, return layer-TGT_SENS output
    text.layers = torch.nn.ModuleList([full[k] for k in range(TGT_SENS + 1)])
    cap.clear(); h = layers[TGT_SENS].register_forward_hook(mk("h"))
    with torch.no_grad(): text(inputs_embeds=batch, use_cache=False)
    h.remove(); text.layers = full
    return cap["h"].float()

acc = {k: {"pred": [], "tgt": [], "tgt_nx": []} for k in ("raw", "scaled", "sanity", "jin")}
for p in load_wikitext_prompts(N_PROMPT + 20)[:N_PROMPT]:
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP + 2: continue
    hent, hL = real_fwd(ids)                       # [S,d]
    emb0 = embed(ids)[0].float()                   # [S,d] layer-0 input
    e = fnorm(hL.to(torch.bfloat16)).float()       # [S,d]  (= e, since W_a=I)
    v = torch.arange(SKIP, S - 1, device=DEV)      # leave room for p+1
    inj = {"raw": e[v],
           "scaled": e[v] * (emb0[v].norm(dim=-1, keepdim=True) / e[v].norm(dim=-1, keepdim=True)),
           "sanity": emb0[v]}
    for name, vec in inj.items():
        batch = emb0.unsqueeze(0).expand(v.numel(), S, D).clone().to(torch.bfloat16)
        batch[torch.arange(v.numel()), v] = vec.to(torch.bfloat16)
        out = lower_fwd(batch)                      # [len(v),S,d]
        pred = out[torch.arange(v.numel()), v]      # read the modified position
        acc[name]["pred"].append(pred); acc[name]["tgt"].append(hent[v]); acc[name]["tgt_nx"].append(hent[v + 1])
    hat = mu["h_ent"] + (e[v] - mu["e"]) @ Jin.T    # J_in baseline, same input/target
    acc["jin"]["pred"].append(hat); acc["jin"]["tgt"].append(hent[v]); acc["jin"]["tgt_nx"].append(hent[v + 1])

C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
print(f"\n{'variant':10s}  uncentered   centered-signal   (vs next-pos target)")
for name in ("sanity", "jin", "raw", "scaled"):
    pr = torch.cat(acc[name]["pred"]); tg = torch.cat(acc[name]["tgt"]); tn = torch.cat(acc[name]["tgt_nx"])
    unc, sig = C(pr, tg), C(pr - mu["h_ent"], tg - mu["h_ent"])
    signx = C(pr - mu["h_ent"], tn - mu["h_ent"])
    print(f"{name:10s}   {unc:.3f}        {sig:.3f}             {signx:.3f}")
print("\nmean baseline (output mu_ent): centered-signal = 0.000 by construction", flush=True)

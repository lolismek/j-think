"""
Decisive re-embed diagnostic (Qwen3-4B, tied). Reuses SAVED W_a / J_in / mu — no refitting.
Question: is the 0.06 end-to-end from W_a (LatentMAS) or from J_in (our sensory bridge)?

For held-out positions we compare three "re-embed" vectors, each pushed to workspace
entry by the SAME J_in, against the real h_ent:
  (1) e_Wa  = fnorm(h_fin) @ W_a      <- what our pipeline does  (tied => W_a~I)
  (2) e_exp = softmax(logits) @ W_in  <- TRUE "align to embedding space" (expected embed)
  (3) emb[t]= embedding of argmax tok  <- hard-token reference (J_in's calibration dist.)
Also: how embedding-like is each (cos to emb[t]), and does W_a==I (cos(e_Wa,fnorm h_fin)).
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka"); MODEL = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"
I_ENTRY = int(os.environ.get("I_ENTRY", 6)); J_EXIT = int(os.environ.get("J_EXIT", 28))
SKIP_FIRST = 16; DEV = "cuda:0"; N_HOLD = int(os.environ.get("N_HOLD", 24)); MAX_LEN = 96
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); layers = model.layers
fnorm = model._final_norm; emb_w = model._embed_tokens.weight.detach().float()
NL, D = model.n_layers, model.d_model; TGT_SENS, FIN = I_ENTRY - 1, NL - 1
print(f"{MODEL} L={NL} d={D} workspace=[{I_ENTRY},{J_EXIT}] TGT_SENS={TGT_SENS}", flush=True)

Wa = torch.load(f"{OUT}/{TAG}_W_a.pt")["W_a"].float().to(DEV)
Jin = torch.load(f"{OUT}/{TAG}_Jin.pt")["Jin"].float().to(DEV)
mu = {k: v.float().to(DEV) for k, v in torch.load(f"{OUT}/{TAG}_stage0_stats.pt")["mu"].items()}

cap = {}
def mk(n):
    def h(m, i, o): cap[n] = (o[0] if isinstance(o, tuple) else o)[0]
    return h
def fwd(ids):
    cap.clear()
    hs = [layers[TGT_SENS].register_forward_hook(mk("ent")), layers[FIN].register_forward_hook(mk("fin"))]
    with torch.no_grad(): model.forward(ids)
    for h in hs: h.remove()

hold = load_wikitext_prompts(N_HOLD + 80)[N_HOLD + 40:]  # disjoint tail from calibration
Hent, Hfin = [], []
for p in hold:
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); v = torch.arange(SKIP_FIRST, S - 1)
    Hent.append(cap["ent"][v].float()); Hfin.append(cap["fin"][v].float())
Hent = torch.cat(Hent).to(DEV); Hfin = torch.cat(Hfin).to(DEV)
N = Hent.shape[0]
cm = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
print(f"held-out positions: {N}", flush=True)

with torch.no_grad():
    fn = fnorm(Hfin.to(torch.bfloat16)).float()                 # post-final-norm hidden
    logits = model.unembed(Hfin.view(N, 1, -1))[:, -1].float()  # [N,V]
    t = logits.argmax(-1); emb_t = emb_w[t]                      # hard token embedding
    p = torch.softmax(logits, -1)
    e_exp = p @ emb_w                                            # expected embedding (true align)
    e_Wa = fn @ Wa                                              # our pipeline's re-embed

def push(e, center):  # J_in sensory bridge
    return mu["h_ent"] + (e - center) @ Jin.T

hat_Wa  = push(e_Wa,  mu["e"])
hat_exp = push(e_exp, e_exp.mean(0))
hat_tok = push(emb_t, mu["b0in"])

print("\n===== IS e IN EMBEDDING SPACE? =====")
print(f"cos(e_Wa, fnorm(h_fin))           = {cm(e_Wa, fn):.3f}   (~1.0 confirms W_a==I: it does NOT remap)")
print(f"cos(e_Wa,  emb[argmax])           = {cm(e_Wa, emb_t):.3f}   (how embedding-like our e is)")
print(f"cos(e_exp, emb[argmax])           = {cm(e_exp, emb_t):.3f}   (expected-embedding: genuinely embed-like)")
print(f"|e_Wa|={e_Wa.norm(dim=-1).mean():.1f}  |e_exp|={e_exp.norm(dim=-1).mean():.1f}  |emb[t]|={emb_t.norm(dim=-1).mean():.2f}  |fnorm|={fn.norm(dim=-1).mean():.1f}")

print("\n===== SAME J_in, DIFFERENT INPUT -> cos(hat, real h_ent) =====")
print(f"J_in( emb[argmax] )  [token, J_in's calib dist]  = {cm(hat_tok, Hent):.3f}")
print(f"J_in( e_exp )        [true embedding-space align] = {cm(hat_exp, Hent):.3f}")
print(f"J_in( e_Wa  )        [OUR PIPELINE]               = {cm(hat_Wa,  Hent):.3f}")
print("DONE", flush=True)

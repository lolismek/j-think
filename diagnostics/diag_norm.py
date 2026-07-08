"""
Is the 0.035 a NORM bug or a DIRECTION problem? (Qwen3-4B tied, saved artifacts.)

The bridge output is  hat = mu_ent + J_in @ (e - mu_e).
Its cosine to real h_ent has TWO parts:
  * common-mode: mu_ent itself is highly correlated with h_ent (they share a mean).
  * signal:      J_in @ (e - mu_e)  -- the only part that actually carries information.
A norm fix can only rescale the signal, which is SCALE-INVARIANT in cosine terms once
we look at the centered (signal-only) alignment. So:
  - centered signal cos(hat - mu_ent, h_ent - mu_ent) isolates real reconstruction.
  - shrinking the e-deviation should RAISE the uncentered cos (defaulting to the mean)
    while leaving the centered signal cos ~unchanged -> proves norms can't fix it.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka"); MODEL = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"
I_ENTRY = int(os.environ.get("I_ENTRY", 6)); SKIP_FIRST = 16; DEV = "cuda:0"
N_HOLD = int(os.environ.get("N_HOLD", 24)); MAX_LEN = 96
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); layers = model.layers
fnorm = model._final_norm; emb_w = model._embed_tokens.weight.detach().float()
NL, D = model.n_layers, model.d_model; TGT_SENS, FIN = I_ENTRY - 1, NL - 1
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

hold = load_wikitext_prompts(N_HOLD + 80)[N_HOLD + 40:]
Hent, Hfin = [], []
for p in hold:
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); v = torch.arange(SKIP_FIRST, S - 1)
    Hent.append(cap["ent"][v].float()); Hfin.append(cap["fin"][v].float())
Hent = torch.cat(Hent).to(DEV); Hfin = torch.cat(Hfin).to(DEV); N = Hent.shape[0]
C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
sig = lambda hat: C(hat - mu["h_ent"], Hent - mu["h_ent"])   # centered signal-only
print(f"held-out positions: {N}", flush=True)

with torch.no_grad():
    fn = fnorm(Hfin.to(torch.bfloat16)).float()
    logits = model.unembed(Hfin.view(N, 1, -1))[:, -1].float()
    t = logits.argmax(-1); emb_t = emb_w[t]
    e_exp = torch.softmax(logits, -1) @ emb_w
    e_Wa = fn @ Wa

push = lambda e, c: mu["h_ent"] + (e - c) @ Jin.T
hat_Wa, hat_exp, hat_tok = push(e_Wa, mu["e"]), push(e_exp, e_exp.mean(0)), push(emb_t, mu["b0in"])

print("\n===== common-mode baseline =====")
print(f"cos(mu_ent, h_ent)                = {C(mu['h_ent'].expand_as(Hent), Hent):.3f}  (just defaulting to the mean)")

print("\n===== UNCENTERED cos(hat, h_ent)  (what the earlier table showed) =====")
print(f"  J_in(e_Wa) [OURS] {C(hat_Wa,Hent):.3f}   J_in(e_exp) {C(hat_exp,Hent):.3f}   J_in(emb) {C(hat_tok,Hent):.3f}")
print("===== CENTERED signal cos(hat-mu, h_ent-mu)  (real reconstruction, norm-invariant) =====")
print(f"  J_in(e_Wa) [OURS] {sig(hat_Wa):.3f}   J_in(e_exp) {sig(hat_exp):.3f}   J_in(emb) {sig(hat_tok):.3f}")

print("\n===== NORM TEST: shrink the e_Wa deviation by factor s (pure norm correction) =====")
for s in [1.0, 0.3, 0.1, 0.03, 0.0]:
    h = mu["h_ent"] + (s * (e_Wa - mu["e"])) @ Jin.T
    print(f"  s={s:<4}: uncentered cos={C(h,Hent):.3f}   centered signal={sig(h):.3f}")
print("(-> uncentered rises to the mean-baseline as s->0; centered signal stays flat & low: NOT a norm bug)")
print("DONE", flush=True)

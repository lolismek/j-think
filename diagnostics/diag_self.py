"""
Decisive: can ANYTHING map a token's OWN embedding -> its OWN workspace-entry state?
Position-consistent (fixes the off-by-one in diag_norm). Compares:
  * mean baseline  : always output mu_ent
  * J_in           : the fitted sensory Jacobian (its literal objective: emb[p] -> h_ent[p])
  * trained ridge  : closed-form linear map emb[p] -> h_ent[p], train/test split (linear ceiling)
Metric that matters = CENTERED signal cos(hat - mu_ent, h_ent - mu_ent): reconstruction beyond the mean.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka"); MODEL = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"
I_ENTRY = int(os.environ.get("I_ENTRY", 6)); SKIP_FIRST = 16; DEV = "cuda:0"
N_HOLD = int(os.environ.get("N_HOLD", 60)); MAX_LEN = 96
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); layers = model.layers
emb_w = model._embed_tokens.weight.detach().float()
NL, D = model.n_layers, model.d_model; TGT_SENS = I_ENTRY - 1
Jin = torch.load(f"{OUT}/{TAG}_Jin.pt")["Jin"].float().to(DEV)
mu = {k: v.float().to(DEV) for k, v in torch.load(f"{OUT}/{TAG}_stage0_stats.pt")["mu"].items()}

cap = {}
def hk(m, i, o): cap["ent"] = (o[0] if isinstance(o, tuple) else o)[0]
def fwd(ids):
    cap.clear(); h = layers[TGT_SENS].register_forward_hook(hk)
    with torch.no_grad(): model.forward(ids)
    h.remove()

hold = load_wikitext_prompts(N_HOLD + 40)
E, Hent = [], []
for p in hold:
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); v = torch.arange(SKIP_FIRST, S - 1)
    E.append(emb_w[ids[0][v]])                    # OWN token embedding at each position
    Hent.append(cap["ent"][v].float())
E = torch.cat(E).to(DEV); Hent = torch.cat(Hent).to(DEV); N = E.shape[0]
C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
sig = lambda hat: C(hat - mu["h_ent"], Hent - mu["h_ent"])
print(f"held-out positions: {N}", flush=True)

# J_in on own token embedding (its fit objective)
hat_jin = mu["h_ent"] + (E - mu["b0in"]) @ Jin.T

# trained ridge emb -> h_ent, train/test split
ntr = N // 2
Xtr, Ytr = E[:ntr] - mu["b0in"], Hent[:ntr] - mu["h_ent"]
lam = 1e-2 * (Xtr.pow(2).sum() / ntr)
W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(D, device=DEV), Xtr.T @ Ytr)
hat_tr = mu["h_ent"] + (E - mu["b0in"]) @ W          # eval on ALL (report test half below too)
sig_tr_test = C((mu["h_ent"] + (E[ntr:] - mu["b0in"]) @ W) - mu["h_ent"], Hent[ntr:] - mu["h_ent"])

print("\n              uncentered    centered-signal")
print(f"mean baseline   {C(mu['h_ent'].expand_as(Hent),Hent):.3f}        0.000")
print(f"J_in            {C(hat_jin,Hent):.3f}        {sig(hat_jin):.3f}   (its own fit objective)")
print(f"trained ridge   {C(hat_tr,Hent):.3f}        {sig(hat_tr):.3f}   (test-half signal={sig_tr_test:.3f})")
print("DONE", flush=True)

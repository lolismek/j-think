"""
Verify the norm-handling fix for the sensory hop. Full pipeline e (motor->reembed).
OLD: h_in = mu_ent + (e-mu_e)@Jin ; then norm-match output to |h_ent|   (rescale AFTER J_in)
NEW: d=(e-mu_e) rescaled to S_CAL (embedding-deviation scale) ; h_in = mu_ent + d@Jin ; NO post-match
Report uncentered cos and CENTERED signal vs real next h_ent (and same-pos h_ent).
Prediction: NEW fixes uncentered (stops being below the mean), centered signal UNCHANGED (~0.04):
            a norm fix cannot change direction.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
import latent_workspace as lw
model, layers = lw.model, lw.layers
mu, Jmot, Wa, Jin, fnorm = lw.mu, lw.Jmot, lw.Wa, lw.Jin, lw.fnorm
emb_w = model._embed_tokens.weight.detach().float()
DEV, SKIP = lw.DEV, 16
J_EXIT, FINAL, TGT_SENS = lw.J_EXIT, lw.FINAL, lw.TGT_SENS
from jlens.examples import load_wikitext_prompts

cap = {}
def mk(n):
    def h(m, i, o): cap[n] = (o[0] if isinstance(o, tuple) else o)[0]
    return h
def fwd(ids):
    cap.clear()
    hs = [layers[J_EXIT].register_forward_hook(mk("hj")), layers[FINAL].register_forward_hook(mk("hfin")),
          layers[TGT_SENS].register_forward_hook(mk("hent"))]
    with torch.no_grad(): model.forward(ids)
    for h in hs: h.remove()

Hj, HL, Hent, Hent_nx, Emb = [], [], [], [], []
for p in load_wikitext_prompts(60):
    ids = model.encode(p, max_length=96); S = ids.shape[1]
    if S <= SKIP + 2: continue
    fwd(ids); v = torch.arange(SKIP, S - 2)
    Hj.append(cap["hj"][v].float()); HL.append(cap["hfin"][v].float())
    Hent.append(cap["hent"][v].float()); Hent_nx.append(cap["hent"][v + 1].float())
    Emb.append(emb_w[ids[0][v]])
Hj, HL = torch.cat(Hj).to(DEV), torch.cat(HL).to(DEV)
Hent, Hent_nx, Emb = torch.cat(Hent).to(DEV), torch.cat(Hent_nx).to(DEV), torch.cat(Emb).to(DEV)

# S_CAL = typical embedding-deviation norm J_in was calibrated on
S_CAL = (Emb - mu["b0in"]).norm(dim=-1).mean()
print(f"S_CAL (embedding-deviation norm) = {S_CAL:.3f}", flush=True)

hatL = mu["h_fin"] + (Hj - mu["h_mot"]) @ Jmot.T
e = fnorm(hatL.to(torch.bfloat16)).float() @ Wa
d = e - mu["e"]

h_old = mu["h_ent"] + d @ Jin.T
h_old = h_old * (Hent_nx.norm(dim=-1, keepdim=True) / h_old.norm(dim=-1, keepdim=True))   # after-match
d_new = d * (S_CAL / d.norm(dim=-1, keepdim=True))
h_new = mu["h_ent"] + d_new @ Jin.T                                                       # no match
h_tok = mu["h_ent"] + (Emb - mu["b0in"]) @ Jin.T                                          # upper bound

C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
sig = lambda hat, tgt: C(hat - mu["h_ent"], tgt - mu["h_ent"])
print(f"\n{'':26s}|h_in|   unc(nx)  signal(nx)   unc(same) signal(same)")
for name, hh in [("OLD (match after J_in)", h_old), ("NEW (rescale before J_in)", h_new),
                 ("token upper-bound", h_tok)]:
    print(f"{name:26s}{hh.norm(dim=-1).mean():6.1f}   {C(hh,Hent_nx):.3f}    {sig(hh,Hent_nx):.3f}       "
          f"{C(hh,Hent):.3f}    {sig(hh,Hent):.3f}")
print(f"\nmean baseline uncentered (mu_ent vs next h_ent) = {C(mu['h_ent'].expand_as(Hent_nx),Hent_nx):.3f}", flush=True)

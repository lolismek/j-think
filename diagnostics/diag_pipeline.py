"""
Trace the EXACT current bridge() hop-by-hop on real states. Import the real engine so
the constants (mu, Jmot, Wa, Jin, fnorm) are IDENTICAL to what the bench uses.
At each hop report: norm, and where a ground truth exists, the CENTERED signal
cos(pred - mu_target, real - mu_target) = how much of the real thought survives.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
import latent_workspace as lw
model, layers, tok = lw.model, lw.layers, lw.tok
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
    hs = [layers[J_EXIT].register_forward_hook(mk("hj")),
          layers[FINAL].register_forward_hook(mk("hfin")),
          layers[TGT_SENS].register_forward_hook(mk("hent"))]
    with torch.no_grad(): model.forward(ids)
    for h in hs: h.remove()

Hj, HL, Hent, Hent_nx, EmbNx = [], [], [], [], []
for p in load_wikitext_prompts(60):
    ids = model.encode(p, max_length=96); S = ids.shape[1]
    if S <= SKIP + 2: continue
    fwd(ids); v = torch.arange(SKIP, S - 2)          # leave room for p+1
    Hj.append(cap["hj"][v].float()); HL.append(cap["hfin"][v].float())
    Hent.append(cap["hent"][v].float()); Hent_nx.append(cap["hent"][v + 1].float())
    EmbNx.append(emb_w[ids[0][v + 1]])
Hj, HL = torch.cat(Hj).to(DEV), torch.cat(HL).to(DEV)
Hent, Hent_nx, EmbNx = torch.cat(Hent).to(DEV), torch.cat(Hent_nx).to(DEV), torch.cat(EmbNx).to(DEV)
N = Hj.shape[0]
C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
n = lambda x: x.norm(dim=-1).mean().item()
cs = lambda pred, real, mu_t: C(pred - mu_t, real - mu_t)   # centered signal

# ---- run the EXACT bridge math, vectorized over positions ----
h = Hj
hatL = mu["h_fin"] + (h - mu["h_mot"]) @ Jmot.T                 # line 1 MOTOR
e = fnorm(hatL.to(torch.bfloat16)).float() @ Wa                # line 2 RE-EMBED
h_in = mu["h_ent"] + (e - mu["e"]) @ Jin.T                     # line 3 SENSORY (pre norm-match)
h_in_nm = h_in * (Hent_nx.norm(dim=-1, keepdim=True) / h_in.norm(dim=-1, keepdim=True))  # line 4

# fidelity check vs the real bridge(): identical up to bf16 storage of bridge()'s output
reld = []
for k in [0, 1, 2, 10, 100]:
    b = lw.bridge(Hj[k], Hent_nx[k].norm()).view(-1).float()
    reld.append(((b - h_in_nm[k]).norm() / b.norm()).item())
    # also: does casting MY float result to bf16 reproduce bridge() exactly?
same_bf16 = torch.equal(lw.bridge(Hj[0], Hent_nx[0].norm()).view(-1),
                        h_in_nm[0].to(torch.bfloat16))
print(f"bridge() fidelity: max rel-diff(float)={max(reld):.4f} (bf16 noise); "
      f"exact-when-both-bf16={same_bf16}", flush=True)

print(f"\nN positions = {N}   (real |h_j|={n(Hj):.0f}  |h_L|={n(HL):.0f}  |h_ent|={n(Hent):.1f})")
print("\n---- hop by hop (centered signal = fraction of the real thought recovered) ----")
print(f"1. MOTOR   hatL   |.|={n(hatL):6.1f}   signal vs real h_L        = {cs(hatL, HL, mu['h_fin']):.3f}")
print(f"2. REEMBED e      |.|={n(e):6.1f}   cos(e, next-token emb)     = {C(e, EmbNx):.3f}   (|emb|={n(EmbNx):.2f}: e is {n(e)/n(EmbNx):.0f}x too big, wrong direction)")
print(f"3. SENSORY h_in   |.|={n(h_in):6.1f}   signal vs real next h_ent  = {cs(h_in, Hent_nx, mu['h_ent']):.3f}")
print(f"4. NORMMATCH h_in |.|={n(h_in_nm):6.1f}   signal vs real next h_ent  = {cs(h_in_nm, Hent_nx, mu['h_ent']):.3f}   <== WHAT THE WORKSPACE ACTUALLY RECEIVES")
print("\n---- control: if the re-embed produced the REAL next-token embedding instead of e ----")
h_in_ctrl = mu["h_ent"] + (EmbNx - mu["b0in"]) @ Jin.T
print(f"   J_in(real next emb) signal vs real next h_ent = {cs(h_in_ctrl, Hent_nx, mu['h_ent']):.3f}   (the 0.33 ceiling; our e destroys it)")
print("DONE", flush=True)

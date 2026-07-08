"""
Stage 0 for Qwen3-4B (single GPU, tied embeddings). Builds W_a, calibration means
(with LOOP-CONSISTENT mu_e), sensory bridge J^in, and corrected diagnostics.
Bounds from env I_ENTRY / J_EXIT.
"""
import os, math, time, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
OUT = os.path.expanduser("~/jlens_cka"); MODEL = "Qwen/Qwen3-4B"; TAG = "qwen3-4b"
I_ENTRY = int(os.environ.get("I_ENTRY", 7)); J_EXIT = int(os.environ.get("J_EXIT", 28))
SKIP_FIRST = 16; DEV = "cuda:0"
N_CAL = int(os.environ.get("N_CAL", 96)); N_HOLD = int(os.environ.get("N_HOLD", 24))
N_FIT = int(os.environ.get("N_FIT", 48)); DIM_BATCH = int(os.environ.get("DIM_BATCH", 64))
MAX_FIT = int(os.environ.get("MAX_FIT", 96)); MAX_LEN = int(os.environ.get("MAX_LEN", 96))
LAM_REL = float(os.environ.get("LAM_REL", 1e-3))
t0 = time.time()
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); text = model._text_module; layers = model.layers
fnorm = model._final_norm; emb_w = model._embed_tokens.weight.detach()
NL, D = model.n_layers, model.d_model; TGT_SENS, FIN = I_ENTRY - 1, NL - 1
print(f"[+{time.time()-t0:.0f}s] {MODEL} L={NL} d={D} workspace=[{I_ENTRY},{J_EXIT}] TGT_SENS={TGT_SENS}", flush=True)
lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", revision="main")
Jmot = lens.jacobians[J_EXIT].float().to(DEV)

# ---------------- W_a (tied: W_out==W_in -> ~I) ----------------
W_out = model._lm_head.weight.detach().float()      # [V,d] (tied to embed)
W_in = emb_w.float()
G = W_out.T @ W_out; lam = LAM_REL * torch.diagonal(G).mean()
Wa = torch.linalg.solve(G + lam * torch.eye(D, device=W_out.device), W_out.T @ W_in)
recon = ((W_out @ Wa - W_in).norm() / W_in.norm()).item()
Wa = Wa.to(DEV); torch.save({"W_a": Wa.cpu(), "recon_rel": recon}, f"{OUT}/{TAG}_W_a.pt")
print(f"[+{time.time()-t0:.0f}s] W_a recon_rel={recon:.4f} (tied->~0), ||W_a-I||={ (Wa-torch.eye(D,device=DEV)).norm().item():.2f}", flush=True)

# ---------------- calibration ----------------
cap = {}
def pre0(m, a, k): cap["b0in"] = a[0]; return None
def mk(n):
    def h(m, i, o): cap[n] = o[0] if isinstance(o, tuple) else o
    return h
def fwd(ids):
    cap.clear()
    hs = [layers[0].register_forward_pre_hook(pre0, with_kwargs=True),
          layers[TGT_SENS].register_forward_hook(mk("ent")), layers[J_EXIT].register_forward_hook(mk("mot")),
          layers[FIN].register_forward_hook(mk("fin"))]
    with torch.no_grad(): model.forward(ids)
    for h in hs: h.remove()
def vidx(S): return torch.arange(SKIP_FIRST, S - 1)

allp = load_wikitext_prompts(N_CAL + N_HOLD + 60)
cal, hold = allp[:N_CAL], allp[N_CAL:N_CAL + N_HOLD]
acc = {k: torch.zeros(D, dtype=torch.float64) for k in ("b0in", "ent", "mot", "fin")}; cnt = 0
MOTS = []
for pi, p in enumerate(cal):
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); v = vidx(S)
    for k in acc: acc[k] += cap[k][0][v].float().sum(0).double().cpu()
    MOTS.append(cap["mot"][0][v].float().cpu()); cnt += int(v.numel())
    if (pi + 1) % 32 == 0: print(f"[+{time.time()-t0:.0f}s]  cal {pi+1}/{len(cal)}", flush=True)
mu = {("h_ent" if k == "ent" else "h_mot" if k == "mot" else "h_fin" if k == "fin" else "b0in"):
      (acc[k] / cnt).float().to(DEV) for k in acc}
# loop-consistent mu_e: mean of e = fnorm(motor_recon) @ W_a
MOTS = torch.cat(MOTS).to(DEV)
hatL = mu["h_fin"] + (MOTS - mu["h_mot"]) @ Jmot.T
mu["e"] = (fnorm(hatL.to(torch.bfloat16)).float() @ Wa).mean(0)
del MOTS, hatL
torch.save({"mu": {k: v.cpu() for k, v in mu.items()}}, f"{OUT}/{TAG}_stage0_stats.pt")
print(f"[+{time.time()-t0:.0f}s] means saved. |mu_e|={mu['e'].norm():.2f} |mu_E|={mu['b0in'].norm():.3f}", flush=True)

# ---------------- fit J^in (truncated) ----------------
full_list = text.layers
setattr(text, "layers", torch.nn.ModuleList([full_list[k] for k in range(TGT_SENS + 1)]))
def fit_jin(prompts):
    Jsum = torch.zeros(D, D, dtype=torch.float32, device=DEV); n = 0
    npass = math.ceil(D / DIM_BATCH); bidx = torch.arange(DIM_BATCH, device=DEV)
    for pi, p in enumerate(prompts):
        ids = model.encode(p, max_length=MAX_FIT); S = ids.shape[1]
        if S <= SKIP_FIRST + 1: continue
        vv = torch.arange(SKIP_FIRST, S - 1, device=DEV); src, tgt = {}, {}
        def pre(m, a, k):
            hh = a[0].detach().requires_grad_(True); src["e"] = hh; return (hh,) + tuple(a[1:]), k
        def th(m, i, o): tgt["t"] = o[0] if isinstance(o, tuple) else o
        ph = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
        thh = layers[TGT_SENS].register_forward_hook(th)
        with torch.enable_grad(): text(input_ids=ids.expand(DIM_BATCH, -1), use_cache=False)
        ph.remove(); thh.remove(); T, e = tgt["t"], src["e"]; cot = torch.zeros_like(T)
        for pp, ds in enumerate(range(0, D, DIM_BATCH)):
            nb = min(DIM_BATCH, D - ds); cot.zero_()
            cot[bidx[:nb, None], vv[None, :], ds + bidx[:nb, None]] = 1.0
            g, = torch.autograd.grad(T, e, grad_outputs=cot, retain_graph=(pp < npass - 1))
            Jsum[ds:ds + nb, :] += g[:nb][:, vv, :].float().mean(dim=1)
        n += 1; del T, e, cot
        if (pi + 1) % 8 == 0: print(f"[+{time.time()-t0:.0f}s]  fit {pi+1}/{len(prompts)}", flush=True)
    return (Jsum / max(n, 1)).cpu(), n
Jin, nfit = fit_jin(allp[:N_FIT]); Jin = Jin.to(DEV)
torch.save({"Jin": Jin.cpu(), "n_prompts": nfit}, f"{OUT}/{TAG}_Jin.pt")
setattr(text, "layers", full_list)
print(f"[+{time.time()-t0:.0f}s] J^in fitted on {nfit} prompts.", flush=True)

# ---------------- diagnostics ----------------
H = {"ent": [], "mot": [], "fin": [], "b0in": []}
for p in hold:
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); v = vidx(S)
    for k in H: H[k].append(cap[k][0][v].float())
H = {k: torch.cat(v).to(DEV) for k, v in H.items()}
cm = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
tok_id = model.unembed(H["fin"]).argmax(-1); emb_t = emb_w[tok_id].float()
raw = H["mot"] @ Jmot.T; aff = mu["h_fin"] + (H["mot"] - mu["h_mot"]) @ Jmot.T
e_real = fnorm(H["fin"].to(torch.bfloat16)).float() @ Wa
e_aff = fnorm(aff.to(torch.bfloat16)).float() @ Wa
hat_e = mu["h_ent"] + (e_real - mu["e"]) @ Jin.T
hat_ea = mu["h_ent"] + (e_aff - mu["e"]) @ Jin.T
hat_tok = mu["h_ent"] + (emb_t - mu["b0in"]) @ Jin.T
print(f"\n===== STAGE-0 DIAGNOSTICS Qwen3-4B (held-out positions: {H['ent'].shape[0]}) =====")
print("[MOTOR]   cos(.,h_fin): raw=%.3f affine=%.3f" % (cm(raw, H["fin"]), cm(aff, H["fin"])))
print("[REEMBED] cos(e,emb[t]): uncentered=%.3f centered=%.3f centered(reconFin)=%.3f"
      % (cm(e_real, emb_t), cm(e_real - mu["e"], emb_t - mu["b0in"]), cm(e_aff - mu["e"], emb_t - mu["b0in"])))
print("[SENSORY] cos(hat_from_token, real h_ent) = %.3f" % cm(hat_tok, H["ent"]))
print("[END2END] cos(hat_from_realFin,  hat_from_token) = %.3f" % cm(hat_e, hat_tok))
print("[END2END] cos(hat_from_reconFin, hat_from_token) = %.3f" % cm(hat_ea, hat_tok))
print(f"[+{time.time()-t0:.0f}s] DONE", flush=True)

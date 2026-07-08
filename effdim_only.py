"""Effective linear dimensionality of the J-lens vectors, per layer.
Lens-only: no model forward passes. eig(J_l^T G J_l), G = centered-unembedding Gram."""
import os, glob, json, subprocess
import numpy as np, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OUT = os.path.expanduser("~/jlens_cka")

def freest_gpu():
    try:
        u = [int(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().split()]
        return min(range(len(u)), key=lambda i: u[i])
    except Exception:
        return 0
g = freest_gpu(); dev = f"cuda:{g}"; torch.cuda.set_device(g); print("using", dev, flush=True)

from huggingface_hub import hf_hub_download
lp = hf_hub_download("neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000")
Jd = torch.load(lp, map_location="cpu", weights_only=False)["J"]
layers = sorted(Jd)

snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))[0]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
key = "model.language_model.embed_tokens.weight"
from safetensors import safe_open
with safe_open(os.path.join(snap, wmap[key]), framework="pt") as f:
    W = f.get_tensor(key).to(torch.float32).to(dev)
Wc = W - W.mean(0, keepdim=True); G = Wc.T @ Wc; del W, Wc
d = G.shape[0]

effdim90, effdim99, pr = [], [], []
for l in layers:
    J = Jd[l].to(torch.float32).to(dev)
    C = J.T @ G @ J; C = 0.5 * (C + C.T)
    ev = torch.linalg.eigvalsh(C).clamp_min(0).flip(0); ev = ev / ev.sum()
    cs = torch.cumsum(ev, 0)
    effdim90.append((int((cs < 0.90).sum()) + 1) / d)
    effdim99.append((int((cs < 0.99).sum()) + 1) / d)
    pr.append(float(ev.sum()**2 / (ev**2).sum()) / d)
layers = np.array(layers)
effdim90, effdim99, pr = map(np.array, (effdim90, effdim99, pr))
np.savez(os.path.join(OUT, "qwen3.5-4b_effdim.npz"),
         layers=layers, effdim90=effdim90, effdim99=effdim99, participation=pr)
print("dims for 90% var (frac):", np.round(effdim90, 3))
print("participation ratio/d :", np.round(pr, 3))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.4, 5))
ax.plot(layers, effdim90, "-o", ms=3.5, label="dims for 90% variance")
ax.plot(layers, effdim99, "-^", ms=3.5, label="dims for 99% variance")
ax.plot(layers, pr, "-s", ms=3.5, label="participation ratio / d")
ax.axvline(2.5, color="k", ls=":", alpha=.5)
ax.text(2.6, ax.get_ylim()[1]*0.9, "sensory|workspace", fontsize=7, color="k")
ax.set_xlabel("source layer"); ax.set_ylabel("fraction of residual-stream dims")
s = ax.secondary_xaxis("top", functions=(lambda x: 100*x/32, lambda r: r*32/100))
s.set_xlabel("relative depth (%)")
ax.set_title("Qwen3.5-4B — effective linear dimensionality of J-lens vectors (lens-only)")
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
p = os.path.join(OUT, "qwen3.5-4b_effdim.png"); fig.savefig(p, dpi=150); print("SAVED:", p)

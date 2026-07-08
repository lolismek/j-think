"""CKA of J-lens vector geometry for Qwen3-4B (36 layers, TIED embeddings) +
auto-detect the 3-region (sensory/workspace/motor) partition. Lens-only, GPU."""
import os, glob, json
import numpy as np, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
OUT = os.path.expanduser("~/jlens_cka"); os.makedirs(OUT, exist_ok=True); TAG = "qwen3-4b"
dev = "cuda:0"; torch.cuda.set_device(0)
from huggingface_hub import hf_hub_download
blob = torch.load(hf_hub_download("neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", revision="main"),
        map_location="cpu", weights_only=False)
Jd = blob["J"]; d = int(blob["d_model"]); layers = sorted(Jd.keys())
print(f"lens: {len(layers)} src layers {layers[0]}..{layers[-1]} d={d}", flush=True)
HUB = os.path.join(os.environ["HF_HOME"], "hub")
snap = glob.glob(os.path.join(HUB, "models--Qwen--Qwen3-4B/snapshots/*"))[0]
idx = os.path.join(snap, "model.safetensors.index.json")
if os.path.exists(idx):
    wmap = json.load(open(idx))["weight_map"]; wu_key = "model.embed_tokens.weight"; f = wmap[wu_key]
else:
    wu_key = "model.embed_tokens.weight"; f = "model.safetensors"
from safetensors import safe_open
with safe_open(os.path.join(snap, f), framework="pt") as h:
    W_U = h.get_tensor(wu_key).to(torch.float32).to(dev)
print("W_U (tied embed)", tuple(W_U.shape), flush=True)
Wc = W_U - W_U.mean(0, keepdim=True); G = (Wc.T @ Wc).float(); del W_U, Wc; torch.cuda.empty_cache()
L = len(layers); J = [Jd[l].float().to(dev) for l in layers]; GJ = [G @ x for x in J]
N = np.zeros((L, L))
for i in range(L):
    JiT = J[i].T
    for j in range(i, L):
        N[i, j] = N[j, i] = float((JiT @ GJ[j]).double().pow(2).sum())
diag = np.sqrt(np.diag(N)); CKA = N / np.outer(diag, diag)
np.save(f"{OUT}/{TAG}_cka.npy", CKA); np.save(f"{OUT}/{TAG}_layers.npy", np.array(layers))

def wmean(a, b):
    s = CKA[a:b, a:b]; n = b - a
    return 1.0 if n < 2 else (s.sum() - n) / (n * n - n)
best = None
for b1 in range(2, 9):
    for b2 in range(b1 + 6, L - 1):
        w = (wmean(0, b1) + wmean(b1, b2) + wmean(b2, L)) / 3
        x = (CKA[0:b1, b1:b2].mean() + CKA[b1:b2, b2:L].mean() + CKA[0:b1, b2:L].mean()) / 3
        if best is None or w - x > best[0]: best = (w - x, b1, b2)
_, b1, b2 = best
NT = L + 1
print(f"AUTO 3-block: sensory L0-L{b1-1} | WORKSPACE L{b1}-L{b2-1} | motor L{b2}-L{layers[-1]}", flush=True)
print(f"  rel-depth: workspace {100*b1/NT:.0f}%-{100*b2/NT:.0f}%   gap(w-x)={best[0]:.3f}", flush=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(CKA, origin="upper", cmap="magma", vmin=CKA.min(), vmax=1.0)
for a, bb, c in [(0, b1, "cyan"), (b1, b2, "lime"), (b2, L, "cyan")]:
    ax.add_patch(Rectangle((a - .5, a - .5), bb - a, bb - a, fill=False, ec=c, lw=2))
ax.set_title(f"Qwen3-4B CKA — workspace L{b1}-L{b2-1}"); ax.set_xlabel("layer"); ax.set_ylabel("layer")
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(f"{OUT}/{TAG}_cka.png", dpi=140)
print("SAVED", f"{OUT}/{TAG}_cka.png", flush=True)

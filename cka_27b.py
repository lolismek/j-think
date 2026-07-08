"""
CKA-over-J-lens-vectors heatmap for Qwen3.6-27B (64 layers, untied embeddings),
same method as the 4B run but GPU-accelerated.

J-lens vectors at layer l = rows of  M_l = W_U @ J_l.  Column-centering M_l over
the token axis reduces the computation to a single [d,d] Gram G = W~_U^T W~_U:

    CKA(i,j) = ||J_i^T G J_j||_F^2 / ( ||J_i^T G J_i||_F * ||J_j^T G J_j||_F )

W_U here is lm_head.weight (Qwen3.6-27B has tie_word_embeddings=False). Lens-only,
no forward passes.
"""
import os, glob, json, subprocess
import numpy as np
import torch

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OUT = os.path.expanduser("~/jlens_cka")
os.makedirs(OUT, exist_ok=True)
TAG = "qwen3.6-27b"

def freest_gpu():
    try:
        u = [int(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().split()]
        return min(range(len(u)), key=lambda i: u[i])
    except Exception:
        return 0
g = freest_gpu(); dev = f"cuda:{g}"; torch.cuda.set_device(g)
print("using", dev, flush=True)

# ---------------- 1. Lens (J_l matrices) ----------------
from huggingface_hub import hf_hub_download
lens_path = hf_hub_download(
    "neuronpedia/jacobian-lens",
    filename="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    revision="main")
blob = torch.load(lens_path, map_location="cpu", weights_only=False)
Jd = blob["J"]
d = int(blob["d_model"])
layers = sorted(Jd.keys())
print(f"lens: {len(layers)} source layers {layers[0]}..{layers[-1]}, d_model={d}, "
      f"n_prompts={blob.get('n_prompts')}", flush=True)

# ---------------- 2. Unembedding W_U = lm_head.weight (untied) ----------------
HUB = os.path.join(os.environ["HF_HOME"], "hub")
snap = glob.glob(os.path.join(HUB, "models--Qwen--Qwen3.6-27B/snapshots/*"))[0]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
wu_key = "lm_head.weight"
from safetensors import safe_open
with safe_open(os.path.join(snap, wmap[wu_key]), framework="pt") as f:
    W_U = f.get_tensor(wu_key).to(torch.float32).to(dev)   # [V, d]
V = W_U.shape[0]
print(f"W_U: {tuple(W_U.shape)} (vocab={V})", flush=True)
assert W_U.shape[1] == d, (W_U.shape, d)

# ---------------- 3. G = centered-unembedding Gram ----------------
Wc = W_U - W_U.mean(0, keepdim=True)
G = (Wc.T @ Wc).to(torch.float32)
del W_U, Wc; torch.cuda.empty_cache()
print(f"G: {tuple(G.shape)} ||G||_F={G.norm().item():.3e}", flush=True)

# ---------------- 4. Pairwise linear CKA ----------------
L = len(layers)
J = [Jd[l].to(torch.float32).to(dev) for l in layers]
GJ = [G @ Jl for Jl in J]
N = np.zeros((L, L), dtype=np.float64)
for i in range(L):
    JiT = J[i].T
    for j in range(i, L):
        C = JiT @ GJ[j]
        val = float((C.double() ** 2).sum())
        N[i, j] = N[j, i] = val
    if (i + 1) % 8 == 0 or i == L - 1:
        print(f"  row {i+1}/{L}", flush=True)

diag = np.sqrt(np.diag(N))
CKA = N / np.outer(diag, diag)
np.save(os.path.join(OUT, f"{TAG}_cka.npy"), CKA)
np.save(os.path.join(OUT, f"{TAG}_layers.npy"), np.array(layers))
print("CKA range:", round(CKA.min(), 3), "..", round(CKA.max(), 3), flush=True)

# ---------------- 5. Heatmap ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 7.4))
im = ax.imshow(CKA, origin="upper", cmap="viridis", vmin=CKA.min(), vmax=1.0)
ax.set_title("Qwen3.6-27B — CKA of J-lens vector geometry across layers", fontsize=11)
ax.set_xlabel("source layer"); ax.set_ylabel("source layer")
ticks = list(range(0, L, 5))
ax.set_xticks(ticks); ax.set_xticklabels([layers[t] for t in ticks])
ax.set_yticks(ticks); ax.set_yticklabels([layers[t] for t in ticks])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="linear CKA")
fig.tight_layout()
png = os.path.join(OUT, f"{TAG}_cka_heatmap.png")
fig.savefig(png, dpi=150); print("SAVED:", png, flush=True)

# ---------------- 6. Adjacent-layer read-out ----------------
adj = np.array([CKA[i, i + 1] for i in range(L - 1)])
print("\nadjacent-layer CKA (boundaries show up as dips):", flush=True)
for i in range(L - 1):
    bar = "#" * int(round(adj[i] * 40))
    print(f"  L{layers[i]:>2}-L{layers[i+1]:<2}  {adj[i]:.3f} {bar}")

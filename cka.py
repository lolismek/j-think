"""
CKA-over-J-lens-vectors heatmap for Qwen3.5-4B, replicating the workspace
block-structure analysis from "Verbalizable Representations Form a Global
Workspace in Language Models" (transformer-circuits.pub/2026/workspace).

J-lens vectors at layer l = rows of  M_l = W_U @ J_l   (one per vocab token).
Per-layer Gram over those vectors is compared across layers with linear CKA.
Column-centering M_l over the token axis reduces the whole computation to a
single [d,d] matrix G = W~_U^T W~_U (W~_U = unembedding centered over vocab):

    CKA(i,j) = ||J_i^T G J_j||_F^2 / ( ||J_i^T G J_i||_F * ||J_j^T G J_j||_F )

No forward passes, no GPU.
"""
import os, glob, json
import numpy as np
import torch

torch.set_num_threads(32)
DEV = "cpu"
OUT = os.path.expanduser("~/jlens_cka")
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. Load the pre-fitted lens (J_l matrices) ----------------
from huggingface_hub import hf_hub_download
lens_path = hf_hub_download(
    "neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000",
)
blob = torch.load(lens_path, map_location="cpu", weights_only=False)
Jd = blob["J"]                       # {layer:int -> [d,d] fp16}
d = int(blob["d_model"])
layers = sorted(Jd.keys())
print(f"lens: {len(layers)} source layers {layers[0]}..{layers[-1]}, d_model={d}, "
      f"n_prompts={blob.get('n_prompts')}")

# ---------------- 2. Unembedding W_U (tied embeddings) ----------------------
snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))[0]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
wu_key = "model.language_model.embed_tokens.weight"
from safetensors import safe_open
with safe_open(os.path.join(snap, wmap[wu_key]), framework="pt") as f:
    W_U = f.get_tensor(wu_key).to(torch.float32)      # [V, d]
V = W_U.shape[0]
print(f"W_U: {tuple(W_U.shape)}  (vocab={V})")
assert W_U.shape[1] == d, (W_U.shape, d)

# ---------------- 3. G = centered-unembedding Gram (once) ------------------
Wc = W_U - W_U.mean(dim=0, keepdim=True)             # center over vocab (token axis)
G = (Wc.T @ Wc).to(torch.float32)                    # [d, d], symmetric PSD
del W_U, Wc
print(f"G: {tuple(G.shape)}  ||G||_F={G.norm().item():.3e}")

# ---------------- 4. Pairwise linear CKA ----------------------------------
L = len(layers)
J = [Jd[l].to(torch.float32) for l in layers]        # list of [d,d]
GJ = [G @ Jl for Jl in J]                             # precompute G @ J_l
# numerator matrix N[i,j] = ||J_i^T G J_j||_F^2 (upper triangle, then mirror)
N = np.zeros((L, L), dtype=np.float64)
for i in range(L):
    JiT = J[i].T
    for j in range(i, L):
        C = JiT @ GJ[j]                               # = J_i^T G J_j  [d,d]
        val = float((C.double() ** 2).sum())
        N[i, j] = N[j, i] = val
    print(f"  row {i+1}/{L} done", flush=True)

diag = np.sqrt(np.diag(N))
CKA = N / np.outer(diag, diag)                        # [L, L], diag == 1

np.save(os.path.join(OUT, "qwen3.5-4b_cka.npy"), CKA)
np.save(os.path.join(OUT, "qwen3.5-4b_layers.npy"), np.array(layers))
print("CKA range:", round(CKA.min(), 3), "..", round(CKA.max(), 3))

# ---------------- 5. Heatmap ----------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(CKA, origin="upper", cmap="viridis", vmin=CKA.min(), vmax=1.0)
ax.set_title("Qwen3.5-4B — CKA of J-lens vector geometry across layers", fontsize=11)
ax.set_xlabel("source layer")
ax.set_ylabel("source layer")
ticks = list(range(0, L, 5))
ax.set_xticks(ticks); ax.set_xticklabels([layers[t] for t in ticks])
ax.set_yticks(ticks); ax.set_yticklabels([layers[t] for t in ticks])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="linear CKA")
fig.tight_layout()
png = os.path.join(OUT, "qwen3.5-4b_cka_heatmap.png")
fig.savefig(png, dpi=150)
print("SAVED:", png)

# ---------------- 6. Quick block read-out ---------------------------------
# adjacent-layer CKA (super-diagonal) highlights block boundaries (dips)
adj = np.array([CKA[i, i + 1] for i in range(L - 1)])
print("\nadjacent-layer CKA (boundaries show up as dips):")
for i in range(L - 1):
    bar = "#" * int(round(adj[i] * 40))
    print(f"  L{layers[i]:>2}-L{layers[i+1]:<2}  {adj[i]:.3f} {bar}")

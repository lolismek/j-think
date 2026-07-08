import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.expanduser("~/jlens_cka")
CKA = np.load(os.path.join(OUT, "qwen3.5-4b_cka.npy"))
layers = np.load(os.path.join(OUT, "qwen3.5-4b_layers.npy"))
L = len(layers)
N_TOTAL = 32  # Qwen3.5-4B text tower depth

# --- objective boundary detection: best single split maximizing block coherence
def block_score(a, b):  # mean off-diagonal CKA within [a,b)
    sub = CKA[a:b, a:b]
    n = b - a
    if n < 2: return 1.0
    return (sub.sum() - n) / (n * n - n)

adj = np.array([CKA[i, i + 1] for i in range(L - 1)])
sensory_end = int(np.argmin(adj[:6]))          # deepest early dip: last sensory layer
print("adjacent-CKA minima (layer, value):",
      sorted(range(L - 1), key=lambda i: adj[i])[:4],
      [round(float(adj[i]), 3) for i in sorted(range(L - 1), key=lambda i: adj[i])[:4]])
print(f"sensory block: L0..L{sensory_end}  (break at L{sensory_end}->L{sensory_end+1})")
print(f"workspace block: L{sensory_end+1}..L{layers[-1]}")
print("mean CKA within sensory:", round(block_score(0, sensory_end + 1), 3))
print("mean CKA within workspace:", round(block_score(sensory_end + 1, L), 3))
print("cross sensory<->workspace:",
      round(float(CKA[:sensory_end + 1, sensory_end + 1:].mean()), 3))

fig, ax = plt.subplots(figsize=(8.6, 7.4))
im = ax.imshow(CKA, origin="upper", cmap="magma", vmin=CKA.min(), vmax=1.0,
               extent=[layers[0] - .5, layers[-1] + .5, layers[-1] + .5, layers[0] - .5])
ax.set_title("Qwen3.5-4B — CKA of J-lens vector geometry across layers\n"
             "(pre-fitted lens, 1000 prompts; source layers 0–30 of 32)", fontsize=10.5)
ax.set_xlabel("source layer"); ax.set_ylabel("source layer")

# mark blocks
b = sensory_end + 0.5
for xy, w in [((layers[0]-.5, layers[0]-.5), sensory_end+1),
              (( b, b), layers[-1]-sensory_end)]:
    ax.add_patch(Rectangle(xy, w, w, fill=False, ec="cyan", lw=2, ls="--"))
ax.text(1, 1, "sensory\n0–%d" % sensory_end, color="cyan", fontsize=8, ha="center", va="center")
mid = (sensory_end + 1 + layers[-1]) / 2
ax.text(mid, mid, "WORKSPACE\nL%d–L%d" % (sensory_end + 1, layers[-1]),
        color="cyan", fontsize=10, ha="center", va="center", fontweight="bold")
# soft internal sub-boundary (workspace core vs output-aligned tail)
sub = sensory_end + 1 + int(np.argmin(adj[sensory_end + 1:])) if L > 10 else None
if sub:
    ax.axhline(sub + 0.5, color="white", lw=1, ls=":", alpha=.6)
    ax.axvline(sub + 0.5, color="white", lw=1, ls=":", alpha=.6)
    print("soft sub-boundary near L%d (workspace core vs output-aligned tail)" % sub)

# top axis: relative depth (paper's 0–100 rescaling)
secx = ax.secondary_xaxis("top", functions=(lambda x: 100 * x / N_TOTAL,
                                            lambda r: r * N_TOTAL / 100))
secx.set_xlabel("relative depth  (layer / 32 × 100)")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10, label="linear CKA")
fig.tight_layout()
png = os.path.join(OUT, "qwen3.5-4b_cka_annotated.png")
fig.savefig(png, dpi=150)
print("SAVED:", png)

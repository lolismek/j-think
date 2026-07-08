"""Annotated CKA heatmap for Qwen3.6-27B: auto-detect the 3-region partition
(sensory / workspace / motor) that maximizes within-block CKA minus cross-block CKA."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.expanduser("~/jlens_cka")
CKA = np.load(os.path.join(OUT, "qwen3.6-27b_cka.npy"))
layers = np.load(os.path.join(OUT, "qwen3.6-27b_layers.npy"))
L = len(layers); N_TOTAL = 64

def wmean(a, b):                       # mean off-diagonal CKA within block [a,b)
    sub = CKA[a:b, a:b]; n = b - a
    return 1.0 if n < 2 else (sub.sum() - n) / (n * n - n)
def cmean(a, b, c, d):                 # mean cross-block CKA
    return float(CKA[a:b, c:d].mean())

best = None
for b1 in range(3, 16):                # sensory|workspace boundary
    for b2 in range(32, 59):           # workspace|motor boundary
        w = (wmean(0, b1) + wmean(b1, b2) + wmean(b2, L)) / 3
        x = (cmean(0, b1, b1, b2) + cmean(b1, b2, b2, L) + cmean(0, b1, b2, L)) / 3
        s = w - x
        if best is None or s > best[0]: best = (s, b1, b2, w, x)
_, b1, b2, w, x = best
print(f"3-block partition: sensory L0-L{b1-1} | workspace L{b1}-L{b2-1} | motor L{b2}-L{layers[-1]}")
print(f"mean within-block CKA = {w:.3f}   cross-block CKA = {x:.3f}   gap = {w-x:.3f}")
print(f"relative depths: sensory 0-{100*b1/N_TOTAL:.0f}% | workspace {100*b1/N_TOTAL:.0f}-{100*b2/N_TOTAL:.0f}% | motor {100*b2/N_TOTAL:.0f}-100%")

fig, ax = plt.subplots(figsize=(9, 7.8))
im = ax.imshow(CKA, origin="upper", cmap="magma", vmin=CKA.min(), vmax=1.0,
               extent=[layers[0]-.5, layers[-1]+.5, layers[-1]+.5, layers[0]-.5])
ax.set_title("Qwen3.6-27B — CKA of J-lens vector geometry across layers\n"
             "(pre-fitted lens, 1000 prompts; source layers 0–62 of 64)", fontsize=10.5)
ax.set_xlabel("source layer"); ax.set_ylabel("source layer")

blocks = [(0, b1, "sensory", "L0–L%d" % (b1-1), 8, False),
          (b1, b2, "WORKSPACE", "L%d–L%d" % (b1, b2-1), 11, True),
          (b2, L, "motor", "L%d–L%d" % (b2, layers[-1]), 9, False)]
for a, bb, name, rng, fs, bold in blocks:
    ax.add_patch(Rectangle((a-.5, a-.5), bb-a, bb-a, fill=False, ec="cyan", lw=2, ls="--"))
    m = (a + bb - 1) / 2
    ax.text(m, m, name + "\n" + rng, color="cyan", fontsize=fs, ha="center", va="center",
            fontweight="bold" if bold else "normal")

secx = ax.secondary_xaxis("top", functions=(lambda v: 100*v/N_TOTAL, lambda r: r*N_TOTAL/100))
secx.set_xlabel("relative depth  (layer / 64 × 100)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10, label="linear CKA")
fig.tight_layout()
png = os.path.join(OUT, "qwen3.6-27b_cka_annotated.png")
fig.savefig(png, dpi=150); print("SAVED:", png)

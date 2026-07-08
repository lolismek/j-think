"""
Fig-28-style workspace onset panel for Qwen3.5-4B:
 (a) next-token prediction accuracy per layer  (top-k J-lens vs model top-1)
 (b) excess kurtosis of J-lens readouts
 (c) top-1 concept autocorrelation across positions (persistence)
 (d) effective linear dimensionality of the J-lens vectors   [lens-only]
"""
import os, glob, json, subprocess
import numpy as np
import torch

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OUT = os.path.expanduser("~/jlens_cka")

N_PROMPTS = int(os.environ.get("N_PROMPTS", 80))
MAX_LEN   = int(os.environ.get("MAX_LEN", 96))
TOPK      = 10
SKIP_FIRST = 4          # drop attention-sink positions (see fitting.py)
LAG = 1                 # autocorrelation lag Δ

def freest_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().split()
        used = [int(x) for x in out]
        return min(range(len(used)), key=lambda i: used[i])
    except Exception:
        return 0

g = freest_gpu(); dev = f"cuda:{g}"; torch.cuda.set_device(g)
print("using", dev, flush=True)

import transformers, jlens
MODEL = "Qwen/Qwen3.5-4B"
try:
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
except Exception:
    hf = transformers.AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
hf = hf.to(dev)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000")
S_LAYERS = lens.source_layers
print("source layers:", S_LAYERS[0], "..", S_LAYERS[-1], flush=True)

# ---------------- corpus ----------------
try:
    from jlens.examples import load_wikitext_prompts
    prompts = load_wikitext_prompts(n_prompts=N_PROMPTS)
    print("corpus: wikitext", len(prompts), flush=True)
except Exception as e:
    print("wikitext unavailable (", type(e).__name__, ") -> built-in corpus", flush=True)
    import textwrap
    base = (
        "The history of the Roman Empire spans over a thousand years of political and "
        "military transformation. Photosynthesis converts sunlight, water and carbon "
        "dioxide into glucose and oxygen inside chloroplasts. In quantum mechanics the "
        "state of a system is described by a wavefunction evolving under the Schrodinger "
        "equation. The novel follows a young cartographer who maps an island that keeps "
        "changing shape overnight. Interest rates set by the central bank ripple through "
        "mortgages, savings and the currency exchange markets. Migratory birds navigate "
        "thousands of miles using the earth magnetic field and the position of the stars. "
        "The recipe calls for slowly caramelizing onions before deglazing the pan with "
        "white wine and stock. Tectonic plates drift a few centimeters each year, building "
        "mountains and opening ocean basins over geological time. The committee debated "
        "the ethics of deploying autonomous systems in densely populated urban centers."
    )
    prompts = [base] * ((N_PROMPTS + 1))
    prompts = prompts[:N_PROMPTS]

# ---------------- accumulators ----------------
acc_hit = {l: 0 for l in S_LAYERS}; acc_n = {l: 0 for l in S_LAYERS}
kurt_sum = {l: 0.0 for l in S_LAYERS}; kurt_n = {l: 0 for l in S_LAYERS}
persist = {l: 0 for l in S_LAYERS}; pairs = {l: 0 for l in S_LAYERS}
top1_all = {l: [] for l in S_LAYERS}     # for null collision baseline

@torch.no_grad()
def excess_kurtosis(x):                    # x: [n, V] fp32 -> [n]
    x = x.double()
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc**2).mean(-1)
    m4 = (xc**4).mean(-1)
    return m4 / var.clamp_min(1e-12)**2 - 3.0

for pi, prompt in enumerate(prompts):
    with torch.no_grad():
        ll, ml, ids = lens.apply(model, prompt, positions=None, max_seq_len=MAX_LEN)
    S = ml.shape[0]
    if S <= SKIP_FIRST + 2:
        continue
    valid = torch.arange(SKIP_FIRST, S - 1, device=ml.device)   # need next token
    model_top1 = ml.argmax(-1)[valid]                            # [nv]
    for l in S_LAYERS:
        Lv = ll[l][valid]                                        # [nv, V] fp32
        # (a) accuracy: model top-1 within lens top-k
        tk = Lv.topk(TOPK, dim=-1).indices
        hit = (tk == model_top1.unsqueeze(-1)).any(-1)
        acc_hit[l] += int(hit.sum()); acc_n[l] += Lv.shape[0]
        # (b) kurtosis
        k = excess_kurtosis(Lv)
        kurt_sum[l] += float(k.sum()); kurt_n[l] += k.numel()
        # (c) persistence (top-1 stable across adjacent valid positions)
        t1 = Lv.argmax(-1)
        persist[l] += int((t1[:-LAG] == t1[LAG:]).sum()); pairs[l] += t1.numel() - LAG
        top1_all[l].append(t1.to("cpu"))
        del Lv
    del ll, ml
    if (pi + 1) % 10 == 0:
        print(f"  {pi+1}/{len(prompts)} prompts", flush=True)

layers = np.array(S_LAYERS)
acc  = np.array([acc_hit[l] / max(acc_n[l], 1) for l in S_LAYERS])
kurt = np.array([kurt_sum[l] / max(kurt_n[l], 1) for l in S_LAYERS])
ac   = np.array([persist[l] / max(pairs[l], 1) for l in S_LAYERS])
# null collision baseline per layer: sum_t p_t^2 over empirical top-1 distribution
null = []
for l in S_LAYERS:
    v = torch.cat(top1_all[l]).numpy()
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    null.append(float((p**2).sum()))
null = np.array(null)

# ---------------- (d) effective linear dimensionality (lens-only) ----------
snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))[0]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
wu_key = "model.language_model.embed_tokens.weight"
from safetensors import safe_open
with safe_open(os.path.join(snap, wmap[wu_key]), framework="pt") as f:
    W_U = f.get_tensor(wu_key).to(torch.float32).to(dev)
Wc = W_U - W_U.mean(0, keepdim=True)
G = Wc.T @ Wc
del W_U, Wc
d = G.shape[0]
effdim90, pr = [], []
for l in S_LAYERS:
    Jl = lens.jacobians[l].to(torch.float32).to(dev)
    C = Jl.T @ G @ Jl                      # covariance of J-lens vectors (up to 1/V)
    C = 0.5 * (C + C.T)
    ev = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)      # desc
    ev = ev / ev.sum()
    n90 = int((torch.cumsum(ev, 0) < 0.90).sum()) + 1
    effdim90.append(n90 / d)
    pr.append(float((ev.sum()**2 / (ev**2).sum())) / d)     # participation ratio / d
effdim90 = np.array(effdim90); pr = np.array(pr)

np.savez(os.path.join(OUT, "qwen3.5-4b_onset_metrics.npz"),
         layers=layers, acc=acc, kurt=kurt, autocorr=ac, null=null,
         effdim90=effdim90, participation=pr)
print("acc", np.round(acc, 3))
print("kurt", np.round(kurt, 1))
print("autocorr", np.round(ac, 3), "null", np.round(null, 3))
print("effdim90", np.round(effdim90, 3))

# ---------------- panel ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
N_TOTAL = 32
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
def depth_axis(ax):
    s = ax.secondary_xaxis("top", functions=(lambda x: 100*x/N_TOTAL,
                                             lambda r: r*N_TOTAL/100))
    s.set_xlabel("relative depth (%)", fontsize=8)
panels = [
    ("(a) next-token accuracy  (model top-1 ∈ J-lens top-%d)" % TOPK, acc, None, "tab:blue"),
    ("(b) excess kurtosis of J-lens readouts", kurt, None, "tab:red"),
    ("(c) top-1 autocorrelation (Δ=%d)  vs null" % LAG, ac, null, "tab:green"),
    ("(d) effective dim (frac. dims for 90%% var)", effdim90, None, "tab:purple"),
]
for ax, (title, y, extra, c) in zip(axes.ravel(), panels):
    ax.plot(layers, y, "-o", ms=3, color=c)
    if extra is not None:
        ax.plot(layers, extra, "--", color="gray", lw=1, label="null")
        ax.legend(fontsize=7)
    ax.axvline(2.5, color="k", ls=":", lw=1, alpha=.5)   # sensory|workspace boundary
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("source layer"); depth_axis(ax)
    ax.grid(alpha=.25)
fig.suptitle("Qwen3.5-4B — workspace onset metrics (J-lens, %d prompts)" % len(prompts),
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, "qwen3.5-4b_onset_panel.png")
fig.savefig(png, dpi=150)
print("SAVED:", png)

"""
Fig-28-style workspace onset panel for Qwen3.6-27B (64 layers):
 (a) next-token accuracy per layer (model top-1 in J-lens top-k)
 (b) excess kurtosis of J-lens readouts
 (c) top-1 concept autocorrelation across positions (persistence) vs null
 (d) effective linear dimensionality of the J-lens vectors  [lens-only]

27B does not fit on one 40GB A100 in bf16, so the model is sharded with
device_map="auto" across the GPUs made visible via CUDA_VISIBLE_DEVICES
(set to 2 idle GPUs by the launcher, leaving the rest free for others).
"""
import os, glob, json, subprocess
import numpy as np
import torch

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # avoid compiled-hook meta path
OUT = os.path.expanduser("~/jlens_cka")
TAG = "qwen3.6-27b"
N_TOTAL = 64
MODEL = "Qwen/Qwen3.6-27B"

N_PROMPTS = int(os.environ.get("N_PROMPTS", 80))
MAX_LEN   = int(os.environ.get("MAX_LEN", 96))
TOPK      = 10
SKIP_FIRST = 4
LAG = 1
CDEV = "cuda:0"          # device for lightweight per-layer readout ops

import transformers, jlens
# Single GPU only (CUDA_VISIBLE_DEVICES pins us to one card). No second GPU -> no
# peer-to-peer / NVLink. Custom device map: keep the unembed hot-path
# (embed/norm/rotary/lm_head + the tiny vision tower) and the first GPU_LAYERS
# transformer layers resident on the GPU; the remaining layers are offloaded and
# streamed GPU<-CPU/disk per forward (host<->device over PCIe, not NVLink).
# offload_folder is REQUIRED, else the offloaded weights are left on the meta device
# ("Cannot copy out of meta tensor").
print("visible GPUs:", torch.cuda.device_count(), flush=True)
cfg = transformers.AutoConfig.from_pretrained(MODEL)
NL = cfg.text_config.num_hidden_layers
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", 38))
dmap = {
    "model.visual": 0,
    "model.language_model.embed_tokens": 0,
    "model.language_model.norm": 0,
    "model.language_model.rotary_emb": 0,
    "lm_head": 0,
}
for i in range(NL):
    dmap[f"model.language_model.layers.{i}"] = 0 if i < GPU_LAYERS else "cpu"
print(f"device map: {GPU_LAYERS}/{NL} layers on GPU, {NL-GPU_LAYERS} streamed from CPU", flush=True)
OFF = "/tmp/aij2115_scratch/offload"; os.makedirs(OFF, exist_ok=True)
_kw = dict(dtype=torch.bfloat16, device_map=dmap, offload_folder=OFF, offload_buffers=True)
try:
    hf = transformers.AutoModelForImageTextToText.from_pretrained(MODEL, **_kw)
except Exception as e:
    print("ImageTextToText load failed (", type(e).__name__, ") -> CausalLM", flush=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, **_kw)
try:
    print("device_map:", {k: str(v) for k, v in hf.hf_device_map.items()
                          if "layers." not in k or k.endswith("layers.0")}, flush=True)
except Exception:
    pass
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    revision="main")
S_LAYERS = lens.source_layers
print("source layers:", S_LAYERS[0], "..", S_LAYERS[-1], "(", len(S_LAYERS), ")", flush=True)

# ---------------- corpus ----------------
try:
    from jlens.examples import load_wikitext_prompts
    prompts = load_wikitext_prompts(n_prompts=N_PROMPTS)
    print("corpus: wikitext", len(prompts), flush=True)
except Exception as e:
    print("wikitext unavailable (", type(e).__name__, ") -> built-in corpus", flush=True)
    base = (
        "The history of the Roman Empire spans over a thousand years of political and "
        "military transformation. Photosynthesis converts sunlight, water and carbon "
        "dioxide into glucose and oxygen inside chloroplasts. In quantum mechanics the "
        "state of a system is described by a wavefunction evolving under the Schrodinger "
        "equation. Interest rates set by the central bank ripple through mortgages, "
        "savings and the currency exchange markets. Tectonic plates drift a few "
        "centimeters each year, building mountains and opening ocean basins over time.")
    prompts = ([base] * N_PROMPTS)[:N_PROMPTS]

# ---------------- accumulators ----------------
acc_hit = {l: 0 for l in S_LAYERS}; acc_n = {l: 0 for l in S_LAYERS}
kurt_sum = {l: 0.0 for l in S_LAYERS}; kurt_n = {l: 0 for l in S_LAYERS}
persist = {l: 0 for l in S_LAYERS}; pairs = {l: 0 for l in S_LAYERS}
top1_all = {l: [] for l in S_LAYERS}

@torch.no_grad()
def excess_kurtosis(x):
    x = x.double()
    xc = x - x.mean(-1, keepdim=True)
    var = (xc ** 2).mean(-1)
    m4 = (xc ** 4).mean(-1)
    return m4 / var.clamp_min(1e-12) ** 2 - 3.0

for pi, prompt in enumerate(prompts):
    with torch.no_grad():
        ll, ml, ids = lens.apply(model, prompt, positions=None, max_seq_len=MAX_LEN)
    S = ml.shape[0]
    if S <= SKIP_FIRST + 2:
        continue
    valid = torch.arange(SKIP_FIRST, S - 1, device=CDEV)
    model_top1 = ml.to(CDEV).argmax(-1)[valid]
    for l in S_LAYERS:
        Lv = ll[l][valid.cpu()].to(CDEV)                    # [nv, V] fp32
        tk = Lv.topk(TOPK, dim=-1).indices
        hit = (tk == model_top1.unsqueeze(-1)).any(-1)
        acc_hit[l] += int(hit.sum()); acc_n[l] += Lv.shape[0]
        k = excess_kurtosis(Lv)
        kurt_sum[l] += float(k.sum()); kurt_n[l] += k.numel()
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
null = []
for l in S_LAYERS:
    v = torch.cat(top1_all[l]).numpy()
    _, c = np.unique(v, return_counts=True); p = c / c.sum()
    null.append(float((p ** 2).sum()))
null = np.array(null)

# free the 27B before the lens-only dimensionality pass
del model, hf, lens
import gc; gc.collect(); torch.cuda.empty_cache()

# ---------------- (d) effective linear dimensionality (lens-only) ----------
from huggingface_hub import hf_hub_download
lp = hf_hub_download("neuronpedia/jacobian-lens",
    filename="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    revision="main")
Jd = torch.load(lp, map_location="cpu", weights_only=False)["J"]
HUB = os.path.join(os.environ["HF_HOME"], "hub")
snap = glob.glob(os.path.join(HUB, "models--Qwen--Qwen3.6-27B/snapshots/*"))[0]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
wu_key = "lm_head.weight"
from safetensors import safe_open
with safe_open(os.path.join(snap, wmap[wu_key]), framework="pt") as f:
    W_U = f.get_tensor(wu_key).to(torch.float32).to(CDEV)
Wc = W_U - W_U.mean(0, keepdim=True); G = Wc.T @ Wc; del W_U, Wc; torch.cuda.empty_cache()
d = G.shape[0]
effdim90, pr = [], []
for l in S_LAYERS:
    Jl = Jd[l].to(torch.float32).to(CDEV)
    C = Jl.T @ G @ Jl; C = 0.5 * (C + C.T)
    ev = torch.linalg.eigvalsh(C).clamp_min(0).flip(0); ev = ev / ev.sum()
    effdim90.append((int((torch.cumsum(ev, 0) < 0.90).sum()) + 1) / d)
    pr.append(float(ev.sum() ** 2 / (ev ** 2).sum()) / d)
    del Jl, C
effdim90 = np.array(effdim90); pr = np.array(pr)

np.savez(os.path.join(OUT, f"{TAG}_onset_metrics.npz"),
         layers=layers, acc=acc, kurt=kurt, autocorr=ac, null=null,
         effdim90=effdim90, participation=pr)
print("acc", np.round(acc, 3), flush=True)
print("kurt", np.round(kurt, 1), flush=True)
print("autocorr", np.round(ac, 3), "null", np.round(null, 3), flush=True)
print("effdim90", np.round(effdim90, 3), flush=True)

# ---------------- panel ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
def depth_axis(ax):
    s = ax.secondary_xaxis("top", functions=(lambda x: 100*x/N_TOTAL,
                                             lambda r: r*N_TOTAL/100))
    s.set_xlabel("relative depth (%)", fontsize=8)
panels = [
    ("(a) next-token accuracy  (model top-1 in J-lens top-%d)" % TOPK, acc, None, "tab:blue"),
    ("(b) excess kurtosis of J-lens readouts", kurt, None, "tab:red"),
    ("(c) top-1 autocorrelation (d=%d) vs null" % LAG, ac, null, "tab:green"),
    ("(d) effective dim (frac. dims for 90%% var)", effdim90, None, "tab:purple"),
]
for ax, (title, y, extra, c) in zip(axes.ravel(), panels):
    ax.plot(layers, y, "-o", ms=3, color=c)
    if extra is not None:
        ax.plot(layers, extra, "--", color="gray", lw=1, label="null"); ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("source layer"); depth_axis(ax); ax.grid(alpha=.25)
fig.suptitle("Qwen3.6-27B — workspace onset metrics (J-lens, %d prompts)" % len(prompts),
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, f"{TAG}_onset_panel.png")
fig.savefig(png, dpi=150); print("SAVED:", png, flush=True)

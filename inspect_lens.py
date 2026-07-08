import os, glob, json
import torch

# ---- 1. Download / locate the pre-fitted Qwen3.5-4B Jacobian lens ----
from huggingface_hub import hf_hub_download
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REV = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
lens_path = hf_hub_download(LENS_REPO, filename=LENS_FILE, revision=LENS_REV)
print("LENS PATH:", lens_path)
print("LENS SIZE (MB):", round(os.path.getsize(lens_path) / 1e6, 1))

obj = torch.load(lens_path, map_location="cpu", weights_only=False)
print("LENS OBJECT TYPE:", type(obj))
if isinstance(obj, dict):
    print("TOP-LEVEL KEYS:", list(obj.keys())[:20], "..." if len(obj) > 20 else "")
    for k, v in list(obj.items())[:6]:
        if torch.is_tensor(v):
            print("   ", k, tuple(v.shape), v.dtype)
        elif isinstance(v, dict):
            print("   ", k, "-> dict with keys", list(v.keys())[:10])
            for kk, vv in list(v.items())[:4]:
                if torch.is_tensor(vv):
                    print("        ", kk, tuple(vv.shape), vv.dtype)
        else:
            print("   ", k, "->", type(v), repr(v)[:120])
else:
    print("ATTRS:", [a for a in dir(obj) if not a.startswith("_")][:40])
    for a in ("layers", "matrices", "jacobians", "J", "state_dict", "n_layers", "d_model"):
        if hasattr(obj, a):
            val = getattr(obj, a)
            print("   attr", a, "->", type(val),
                  tuple(val.shape) if torch.is_tensor(val) else "")

# ---- 2. Locate cached Qwen3.5-4B model + its unembedding layout ----
print("\n=== MODEL ===")
snaps = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))
print("SNAPSHOTS:", snaps)
snap = snaps[0]
cfg = json.load(open(os.path.join(snap, "config.json")))
for k in ("model_type", "hidden_size", "vocab_size", "num_hidden_layers",
          "tie_word_embeddings"):
    print("   cfg", k, "=", cfg.get(k))

# safetensors weight map -> which keys hold the (un)embedding
idx_path = os.path.join(snap, "model.safetensors.index.json")
if os.path.exists(idx_path):
    wmap = json.load(open(idx_path))["weight_map"]
    hits = {k: v for k, v in wmap.items()
            if "lm_head" in k or "embed_tokens" in k}
    print("   EMBED/HEAD KEYS:", hits)
else:
    from safetensors import safe_open
    st = os.path.join(snap, "model.safetensors")
    with safe_open(st, framework="pt") as f:
        keys = [k for k in f.keys() if "lm_head" in k or "embed_tokens" in k]
        print("   single-file keys:", keys)
        for k in keys:
            print("       ", k, f.get_slice(k).get_shape())

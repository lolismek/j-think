"""Download Qwen3.6-27B weights + its Jacobian lens into aij2115's own HF cache."""
import os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import snapshot_download, hf_hub_download

t0 = time.time()
print("== downloading lens .pt ==", flush=True)
lp = hf_hub_download(
    "neuronpedia/jacobian-lens",
    filename="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    revision="main")
print("LENS:", lp, "(%.0fs)" % (time.time() - t0), flush=True)

print("== downloading model weights (15 shards) ==", flush=True)
mp = snapshot_download(
    "Qwen/Qwen3.6-27B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    max_workers=8)
print("MODEL:", mp, "(%.0fs)" % (time.time() - t0), flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)

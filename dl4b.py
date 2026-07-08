import os
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
from huggingface_hub import snapshot_download, hf_hub_download
print("MODEL", snapshot_download("Qwen/Qwen3-4B"), flush=True)
print("LENS", hf_hub_download("neuronpedia/jacobian-lens",
      filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", revision="main"), flush=True)
print("DL_DONE", flush=True)

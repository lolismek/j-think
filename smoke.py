import os, subprocess, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def freest_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().split()
        used = [int(x) for x in out]
        return min(range(len(used)), key=lambda i: used[i])
    except Exception:
        return 0

g = freest_gpu()
dev = f"cuda:{g}"
torch.cuda.set_device(g)
print("using", dev)

import transformers, jlens
try:
    import datasets
    print("datasets:", datasets.__version__)
except Exception as e:
    print("datasets: MISSING", type(e).__name__)

MODEL = "Qwen/Qwen3.5-4B"
try:
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
except Exception as e:
    print("AutoModelForCausalLM failed:", type(e).__name__, "-> trying ImageTextToText")
    hf = transformers.AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
hf = hf.to(dev)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
print("n_layers:", model.n_layers, "d_model:", model.d_model)

lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000",
)
print(repr(lens))

ll, ml, ids = lens.apply(model, "The capital of France is", positions=[-1])
ks = sorted(ll)
print("n_positions x vocab:", tuple(ll[ks[0]].shape))
for l in [ks[0], ks[len(ks)//2], ks[-1]]:
    toks = [tok.decode([t]) for t in ll[l][0].topk(3).indices.tolist()]
    print(f"  L{l:>2} J-lens:", toks)
print("  model :", [tok.decode([t]) for t in ml[0].topk(3).indices.tolist()])

# corpus check
try:
    from jlens.examples import load_wikitext_prompts
    ps = load_wikitext_prompts(n_prompts=3)
    print("wikitext prompts ok:", len(ps), "example len(chars):", len(ps[0]))
except Exception as e:
    print("load_wikitext_prompts FAILED:", type(e).__name__, str(e)[:120])
print("SMOKE_OK")

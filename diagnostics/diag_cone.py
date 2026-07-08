"""
Where does the '0.6' come from? Answer: the workspace-entry states all live in a
narrow cone around their mean, so a CONSTANT vector (the mean) already scores ~0.6
against any of them -- WITHOUT looking at the input at all. The information that
distinguishes one thought from another is the small deviation off that mean.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
MODEL = "Qwen/Qwen3-4B"; I_ENTRY = int(os.environ.get("I_ENTRY", 6)); TGT_SENS = I_ENTRY - 1
SKIP_FIRST = 16; DEV = "cuda:0"; MAX_LEN = 96
import transformers, jlens
from jlens.examples import load_wikitext_prompts

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()
model = jlens.from_hf(hf, tok); layers = model.layers
cap = {}
def hk(m, i, o): cap["ent"] = (o[0] if isinstance(o, tuple) else o)[0]
def fwd(ids):
    h = layers[TGT_SENS].register_forward_hook(hk)
    with torch.no_grad(): model.forward(ids)
    h.remove()
H = []
for p in load_wikitext_prompts(60):
    ids = model.encode(p, max_length=MAX_LEN); S = ids.shape[1]
    if S <= SKIP_FIRST + 1: continue
    fwd(ids); H.append(cap["ent"][torch.arange(SKIP_FIRST, S - 1)].float())
H = torch.cat(H).to(DEV); N = H.shape[0]
mu = H.mean(0); delta = H - mu
C = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)

print(f"N workspace-entry states = {N}")
print(f"\n|h_ent| (each state)      mean = {H.norm(dim=-1).mean():.1f}")
print(f"|mu_ent| (the mean vec)          = {mu.norm():.1f}")
print(f"|h_ent - mu_ent| (deviation) mean = {delta.norm(dim=-1).mean():.1f}")
print(f"\n=> energy in the common mean = {mu.norm()**2 / (H.norm(dim=-1)**2).mean() * 100:.0f}%  of each state")
print(f"cos(mu_ent, h_ent)          mean = {C(mu.expand_as(H), H).mean():.3f}   <-- THE '0.6', from a CONSTANT vector")
idx = torch.randperm(N, device=DEV)
print(f"cos(h_ent_i, h_ent_j) random pairs= {C(H, H[idx]).mean():.3f}   <-- two DIFFERENT thoughts already this aligned")
print(f"cos(delta_i, delta_j) random pairs= {C(delta, delta[idx]).mean():.3f}   <-- the token-specific part IS ~orthogonal")
print("\n=> 0.6 = blind guessing the average. Real info lives in the deviation, which is what 'centered signal' measures.")

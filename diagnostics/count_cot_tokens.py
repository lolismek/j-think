"""
How many tokens does vanilla Qwen3-4B write between the prompt and the answer on
GSM8K CoT? This is the natural scale for K (each latent step ~ one 'thought').

Greedy-decode the SAME few-shot CoT prompt the bench uses. For each problem count
the generated tokens up to the "The answer is" marker (the reasoning span). Report
the distribution so we can pick K to match.
"""
import os, re
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
import torch, transformers

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
N_PROB = int(os.environ.get("N_PROB", 50))
DEV = "cuda:0"

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}); hf.eval()

FEWSHOT = """Solve each math problem step by step, then give the final answer as "The answer is N."

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
A: In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72 clips. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of babysitting. How much did she earn?
A: 50 minutes is 50 / 60 of an hour. She earned 12 * 50 / 60 = 10 dollars. The answer is 10.
"""

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    out = []
    for r in ds:
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append((r["question"].strip(), gold))
    return out

def parse_num(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None

@torch.no_grad()
def run(prompt_ids):
    out = hf.generate(prompt_ids, max_new_tokens=300, do_sample=False,
                      pad_token_id=tok.eos_token_id)[0, prompt_ids.shape[1]:]
    txt = tok.decode(out, skip_special_tokens=True).split("\nQ:")[0]
    # reasoning span = everything up to (and not including) "The answer is"
    if "The answer is" in txt:
        reasoning = txt.split("The answer is")[0]
        found_marker = True
    else:
        reasoning = txt
        found_marker = False
    # count reasoning tokens as re-tokenized (add_special_tokens=False)
    n_reason = len(tok(reasoning, add_special_tokens=False).input_ids)
    n_total  = out.shape[0]
    seg = txt.split("The answer is")[-1] if found_marker else txt
    return n_reason, n_total, found_marker, parse_num(seg)

probs = load_gsm8k(N_PROB)
reason_lens, total_lens, no_marker, correct = [], [], 0, 0
print(f"{MODEL}  GSM8K CoT reasoning-length probe, {len(probs)} problems\n", flush=True)
for qi, (q, gold) in enumerate(probs):
    prompt = FEWSHOT + f"\nQ: {q}\nA:"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    nr, nt, fm, pred = run(ids)
    reason_lens.append(nr); total_lens.append(nt)
    if not fm: no_marker += 1
    try: ok = pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except ValueError: ok = (pred == gold)
    if ok: correct += 1
    print(f"[{qi+1}/{len(probs)}] reason_tok={nr:3d}  total_tok={nt:3d}  {'no-marker ' if not fm else ''}pred={pred} gold={gold} {'✓' if ok else ''}", flush=True)

t = torch.tensor(reason_lens, dtype=torch.float)
q = torch.tensor([0.1,0.25,0.5,0.75,0.9])
qs = torch.quantile(t, q).tolist()
print(f"\n===== reasoning tokens (prompt -> 'The answer is') over {len(reason_lens)} problems =====")
print(f"  mean={t.mean():.1f}  std={t.std():.1f}  min={int(t.min())}  max={int(t.max())}")
print(f"  quantiles  p10={qs[0]:.0f}  p25={qs[1]:.0f}  median={qs[2]:.0f}  p75={qs[3]:.0f}  p90={qs[4]:.0f}")
print(f"  problems with no 'The answer is' marker in 300 tok: {no_marker}")
print(f"  CoT accuracy: {correct}/{len(probs)} = {correct/len(probs):.2f}")
print(f"\n=> suggested K ~ median reasoning length = {qs[2]:.0f}", flush=True)

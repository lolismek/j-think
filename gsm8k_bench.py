"""
Stage 2: mini GSM8K bench on Qwen3.6-27B — latent-workspace regime vs normal regime.
Imports the engine from latent_workspace (loads model once, defines latent_generate).

Regimes (same model, same few-shot prompt):
  * normal-CoT   : model generates text reasoning then "The answer is N."   (hf.generate)
  * normal-direct: K=0 latent steps, cue "The answer is" -> decode number    (no thinking)
  * latent-K     : K workspace latent steps, cue "The answer is" -> number    (our method)
"""
import os, re, sys
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
import torch
import latent_workspace as lw
tok, hf, DEV = lw.tok, lw.hf, lw.DEV

N_PROB = int(os.environ.get("N_PROB", 16))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "4,8").split(",")]

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

def gold_eq(pred, gold):
    if pred is None: return False
    try: return abs(float(pred) - float(gold)) < 1e-4
    except ValueError: return pred == gold

cue_ids = tok(" The answer is", add_special_tokens=False).input_ids
stop_ids = tok("\n", add_special_tokens=False).input_ids

@torch.no_grad()
def normal_cot(prompt_ids):
    out = hf.generate(prompt_ids, max_new_tokens=256, do_sample=False,
                      pad_token_id=tok.eos_token_id)[0, prompt_ids.shape[1]:]
    txt = tok.decode(out, skip_special_tokens=True).split("\nQ:")[0]
    seg = txt.split("The answer is")[-1] if "The answer is" in txt else txt
    return parse_num(seg), txt.strip()[:160]

@torch.no_grad()
def cued(prompt_ids, K):
    ids = lw.latent_generate(prompt_ids, K, cue_ids, max_new=12, stop_ids=stop_ids)
    txt = tok.decode(ids, skip_special_tokens=True)
    return parse_num(txt), txt.strip()[:60]

def main():
    assert lw.selftest(), "runtime self-test failed"
    probs = load_gsm8k(N_PROB)
    methods = ["cot", "K0"] + [f"K{k}" for k in K_LIST]
    correct = {m: 0 for m in methods}
    print(f"\n===== GSM8K mini-bench ({len(probs)} problems), {lw.MODEL} workspace[{lw.I_ENTRY},{lw.J_EXIT}] =====", flush=True)
    for qi, (q, gold) in enumerate(probs):
        prompt = FEWSHOT + f"\nQ: {q}\nA:"
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        pc, ct = normal_cot(ids)
        p0, t0 = cued(ids, 0)
        res = {"cot": pc, "K0": p0}
        for k in K_LIST: res[f"K{k}"], _ = cued(ids, k)
        for m in methods:
            if gold_eq(res[m], gold): correct[m] += 1
        line = "  ".join(f"{m}={res[m]}" for m in methods)
        print(f"[{qi+1}/{len(probs)}] gold={gold} | {line}  {'CoT✓' if gold_eq(pc,gold) else ''}", flush=True)
    print("\n===== ACCURACY =====")
    for m in methods:
        print(f"  {m:5s}: {correct[m]}/{len(probs)} = {correct[m]/len(probs):.2f}")

if __name__ == "__main__":
    main()

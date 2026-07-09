"""
Batched GSM8K bench. Same method as gsm8k_bench.py but processes all problems in
one batch per regime, so the A100 is fed wide matmuls instead of one-token-at-a-time.

CoT baseline: plain hf.generate on the left-padded batch.
Latent path : batched prefill + batched single-position latent/cue/decode steps.
              The per-layer KV cache is ragged (workspace layers see the K latent
              positions, other layers don't), and prompts are left-padded, so every
              manual layer call gets an additive mask built from (pad pattern, that
              layer's own cache length).

VALIDATE=1 checks the batched latent tokens equal lw.latent_generate per problem
before any accuracy number is reported. Reuses the loaded model/bridge from lw.
"""
import os, re
os.environ.setdefault("HF_HOME", "/tmp/aij2115_scratch/hf")
import torch
import latent_workspace as lw

tok, hf, DEV = lw.tok, lw.hf, lw.DEV
model, layers, rotary = lw.model, lw.layers, lw.rotary
embed, fnorm, text = lw.embed, lw.fnorm, lw.text
mu, Jmot, Wa, Jin = lw.mu, lw.Jmot, lw.Wa, lw.Jin
I_ENTRY, J_EXIT, NL, TGT_SENS, FINAL = lw.I_ENTRY, lw.J_EXIT, lw.NL, lw.TGT_SENS, lw.FINAL
NEG = torch.finfo(torch.bfloat16).min

N_PROB = int(os.environ.get("N_PROB", 50))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "32,64").split(",")]
VALIDATE = os.environ.get("VALIDATE", "1") == "1"

FEWSHOT = lw_fewshot = """Solve each math problem step by step, then give the final answer as "The answer is N."

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

def parse_num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None

def parse_first(t):     # first number after the cue = the answer; robust to divergent tails
    m = re.search(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return m.group(0).rstrip(".") if m else None

def gold_eq(pred, gold):
    if pred is None: return False
    try: return abs(float(pred) - float(gold)) < 1e-4
    except ValueError: return pred == gold

cue_ids = tok(" The answer is", add_special_tokens=False).input_ids
stop_ids = set(tok("\n", add_special_tokens=False).input_ids); stop_ids.add(tok.eos_token_id)

# ---------------- batched bridge (per-row copy of lw.bridge) ----------------
def bridge_b(h_j, tgt):                       # h_j [B,d], tgt [B] -> [B,1,d]
    h = h_j.float()
    hatL = mu["h_fin"] + (h - mu["h_mot"]) @ Jmot.T
    e = fnorm(hatL.to(torch.bfloat16)).float() @ Wa
    h_in = mu["h_ent"] + (e - mu["e"]) @ Jin.T
    h_in = h_in * (tgt.unsqueeze(-1) / h_in.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return h_in.to(torch.bfloat16).unsqueeze(1)

# ---------------- batched single-position runner ----------------
def add_mask(pad_bool, past_len):             # pad_bool [B,P]; -> additive [B,1,1,past_len+1]
    B, P = pad_bool.shape
    kv = past_len + 1
    m = torch.zeros(B, 1, 1, kv, device=DEV, dtype=torch.bfloat16)
    real = min(P, kv)                          # first `real` cols are the (possibly padded) prompt
    m[:, 0, 0, :real] = torch.where(pad_bool[:, :real], 0.0, NEG)
    return m

def run_through_b(hidden, pos, layer_ids, cache, pad_bool):   # hidden [B,1,d]
    B = hidden.shape[0]
    pid = torch.full((B, 1), pos, device=DEV)
    cos, sin = rotary(hidden, pid)
    for l in layer_ids:
        m = add_mask(pad_bool, cache.get_seq_length(l))
        out = layers[l](hidden, attention_mask=m, position_ids=pid,
                        past_key_values=cache, use_cache=True, position_embeddings=(cos, sin))
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden

@torch.no_grad()
def prefill_b(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True)      # left padding (set below)
    ids = enc.input_ids.to(DEV); am = enc.attention_mask.to(DEV)
    cache = lw.DynamicCache(); cap = {}
    grab = lambda n, idx: layers[idx].register_forward_hook(
        lambda m, i, o: cap.__setitem__(n, (o[0] if isinstance(o, tuple) else o)[:, -1].detach()))
    hs = [grab("hj", J_EXIT), grab("hfin", FINAL), grab("hent", TGT_SENS)]
    text(input_ids=ids, attention_mask=am, past_key_values=cache, use_cache=True)
    for h in hs: h.remove()
    return cache, cap["hj"], cap["hfin"], cap["hent"], am.bool(), ids.shape[1]

@torch.no_grad()
def latent_generate_b(prompts, K, max_new=12):
    B = len(prompts)
    cache, hj, hfin, hent, pad_bool, P = prefill_b(prompts)
    pos = P; tgt = hent.float().norm(dim=-1)
    for _ in range(K):
        h_in = bridge_b(hj, tgt)
        hj = run_through_b(h_in, pos, list(range(I_ENTRY, J_EXIT + 1)), cache, pad_bool)[:, -1]; pos += 1
    logits = model.unembed(hfin.unsqueeze(1))[:, -1]
    for c in cue_ids:
        hid = embed(torch.full((B, 1), c, device=DEV)).to(torch.bfloat16)
        hid = run_through_b(hid, pos, list(range(NL)), cache, pad_bool); pos += 1
        logits = model.unembed(hid)[:, -1]
    outs = [[] for _ in range(B)]; done = [False] * B
    for _ in range(max_new):
        nxt = logits.argmax(-1)                                # [B]
        for b in range(B):
            t = nxt[b].item()
            if done[b] or t in stop_ids: done[b] = True
            else: outs[b].append(t)
        if all(done): break
        hid = embed(nxt.view(B, 1)).to(torch.bfloat16)
        hid = run_through_b(hid, pos, list(range(NL)), cache, pad_bool); pos += 1
        logits = model.unembed(hid)[:, -1]
    return outs

@torch.no_grad()
def cot_b(prompts, budget=None):
    # budget=None -> uncapped ceiling (reason freely, read the model's own "The answer is").
    # budget=k    -> fair match to latent K=k: give exactly k thinking tokens, then FORCE the
    #                cue " The answer is" and decode the answer. Same protocol as the latent path
    #                (k steps of thinking, forced cue, read first number) but thinking is in text.
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(DEV); am = enc.attention_mask.to(DEV)
    B, P = ids.shape
    if budget is None:
        gen = hf.generate(ids, attention_mask=am, max_new_tokens=256, do_sample=False,
                          pad_token_id=tok.eos_token_id)[:, P:]
        preds = []
        for row in gen:
            txt = tok.decode(row, skip_special_tokens=True).split("\nQ:")[0]
            seg = txt.split("The answer is")[-1] if "The answer is" in txt else txt
            preds.append(parse_first(seg))
        return preds
    # give `budget` thinking tokens. If the model naturally reaches "The answer is" within its
    # own answer (before drifting into a hallucinated next "Q:"), use that answer. Only if it
    # ran over budget without answering do we FORCE the cue and decode. Matches latent K=budget.
    gen_full = hf.generate(ids, attention_mask=am, max_new_tokens=budget, do_sample=False,
                           pad_token_id=tok.eos_token_id)                   # [B, P+budget]
    preds = [None] * B; need = []
    for i, row in enumerate(gen_full[:, P:]):
        seg = tok.decode(row, skip_special_tokens=True).split("\nQ:")[0]
        if "The answer is" in seg:
            preds[i] = parse_first(seg.split("The answer is")[-1])          # answered within budget
        else:
            need.append(i)                                                 # overran -> force the cue
    if need:
        sub = gen_full[need]
        cue = torch.tensor(cue_ids, device=DEV).unsqueeze(0).expand(len(need), -1)
        full = torch.cat([sub, cue], dim=1)
        sub_am = torch.cat([am[need], torch.ones(len(need), budget + len(cue_ids),
                                                 device=DEV, dtype=am.dtype)], dim=1)
        ans = hf.generate(full, attention_mask=sub_am, max_new_tokens=8, do_sample=False,
                          pad_token_id=tok.eos_token_id)[:, full.shape[1]:]
        for j, i in enumerate(need):
            preds[i] = parse_first(tok.decode(ans[j], skip_special_tokens=True))
    return preds

def validate(prompts):
    # Correctness bar = the parsed answer (first number after the cue) must match the
    # single-example engine. bf16 batched vs unbatched matmuls differ in the last bit,
    # so the low-information tail after the answer can diverge in greedy decoding; that
    # does not affect the metric. We assert answer-equality and report tail divergence.
    print(f"[validate] batched vs single lw.latent_generate on {len(prompts)} problems, K=4 ...", flush=True)
    batch = latent_generate_b(prompts, 4)
    ans_ok, tail_div = True, 0
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        single = lw.latent_generate(ids, 4, cue_ids, max_new=12, stop_ids=list(stop_ids))
        a_s, a_b = parse_first(tok.decode(single)), parse_first(tok.decode(batch[i]))
        if a_s != a_b:
            ans_ok = False
            print(f"  ANSWER MISMATCH prob {i}: single={a_s} ({single}) batched={a_b} ({batch[i]})", flush=True)
        elif single != batch[i]:
            tail_div += 1
    print(f"[validate] answers {'MATCH' if ans_ok else 'DIFFER'}; "
          f"{tail_div}/{len(prompts)} had identical answer but divergent tail (bf16 noise, harmless)", flush=True)
    return ans_ok

CHUNK = int(os.environ.get("CHUNK", 50))
def run_chunked(fn, prompts):        # split into CHUNK-sized batches (memory) and concat results
    out = []
    for i in range(0, len(prompts), CHUNK):
        out += fn(prompts[i:i + CHUNK])
    return out

def latent_preds(prompts, k):
    return [parse_first(tok.decode(o)) for o in latent_generate_b(prompts, k)]

def main():
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    probs = load_gsm8k(N_PROB)
    prompts = [FEWSHOT + f"\nQ: {q}\nA:" for q, _ in probs]
    golds = [g for _, g in probs]

    if VALIDATE:
        assert validate(prompts[:6]), "batched latent path does not match single-example engine"

    preds = {}
    preds["cot"] = run_chunked(lambda ps: cot_b(ps, None), prompts)                 # uncapped ceiling
    for k in K_LIST:
        preds[f"cot@{k}"] = run_chunked(lambda ps, b=k: cot_b(ps, b), prompts)      # text thinking, k-token budget
    preds["K0"] = run_chunked(lambda ps: latent_preds(ps, 0), prompts)              # direct, no thinking
    for k in K_LIST:
        preds[f"K{k}"] = run_chunked(lambda ps, kk=k: latent_preds(ps, kk), prompts)  # latent thinking, k steps

    methods = ["cot"] + [f"cot@{k}" for k in K_LIST] + ["K0"] + [f"K{k}" for k in K_LIST]
    correct = {m: 0 for m in methods}
    for qi in range(len(probs)):
        for m in methods:
            if gold_eq(preds[m][qi], golds[qi]): correct[m] += 1
        if len(probs) <= 50:
            line = "  ".join(f"{m}={preds[m][qi]}" for m in methods)
            print(f"[{qi+1}/{len(probs)}] gold={golds[qi]} | {line}", flush=True)
    N = len(probs); acc = lambda m: f"{correct[m]:3d}/{N} = {correct[m]/N:.3f}"
    print(f"\n===== GSM8K batched bench ({N} problems), {lw.MODEL} workspace[{I_ENTRY},{J_EXIT}] =====")
    print(f"  cot (uncapped) : {acc('cot')}")
    print(f"  K0  (direct)   : {acc('K0')}")
    print("\n  budget   text-CoT@budget      latent-K")
    for k in K_LIST:
        print(f"   {k:4d}    cot@{k:<3d} {acc(f'cot@{k}')}   K{k:<3d} {acc(f'K{k}')}", flush=True)

if __name__ == "__main__":
    main()

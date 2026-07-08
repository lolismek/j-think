# Workspace CKA — Qwen3.5-4B

Locating the **global workspace** layer range in Qwen3.5-4B by replicating the
CKA block-structure analysis from *"Verbalizable Representations Form a Global
Workspace in Language Models"* (transformer-circuits.pub/2026/workspace) using
the Jacobian lens (github.com/anthropics/jacobian-lens).

## Method

J-lens vectors at layer `l` are the **rows of `W_U J_l`** (one direction per
vocab token). The paper compares, for each layer pair, the Gram matrix of
pairwise similarities among these vectors, via linear CKA. Column-centering
`M_l = W_U J_l` over the token axis collapses the whole thing to a single
`[d,d]` matrix `G = W̃_Uᵀ W̃_U` (centered-unembedding Gram), so:

```
CKA(i,j) = ‖J_iᵀ G J_j‖_F² / ( ‖J_iᵀ G J_i‖_F · ‖J_jᵀ G J_j‖_F )
```

No forward passes, no GPU — just the pre-fitted lens + the tied unembedding.

## Inputs

- **Model:** `Qwen/Qwen3.5-4B` (multimodal; text tower = 32 layers, d=2560,
  vocab=248320; embeddings tied → `W_U = model.language_model.embed_tokens.weight`).
- **Lens:** `neuronpedia/jacobian-lens`, revision `qwen-n1000`, file
  `qwen3.5-4b/.../Qwen3.5-4B_jacobian_lens_n1000.pt` (fit on 1000 prompts,
  source layers 0–30).

## Scripts

| file | what |
|---|---|
| `probe.py` | check torch/transformers/etc. in a conda env |
| `inspect_lens.py` | download lens, dump its structure + model unembedding layout |
| `cka.py` | compute the CKA matrix + heatmap (`results/`) |
| `annotate.py` | block-boundary detection + annotated heatmap |

Run in an env with torch + transformers + safetensors + matplotlib
(on tigerfish: conda env `2dtf`), e.g.:

```bash
export HF_HOME=$HOME/.cache/huggingface
python cka.py        # -> results/qwen3.5-4b_cka{.npy,_heatmap.png}
python annotate.py   # -> results/qwen3.5-4b_cka_annotated.png
```

## Result

Clear three-region signature, matching the paper:

| region | layers | note |
|---|---|---|
| sensory / early | **L0–L2** | within-CKA 0.98, cross-to-rest 0.47; sharp break at L2→L3 (0.777) |
| **workspace / middle** | **L3–L30** | one long high-CKA block; brightest core ≈ L6–L18 |
| motor / late | (L31) | final layer excluded from lens (top-layer Jacobian ≈ identity) |

**Candidate workspace range: ~L3–L30 of 32** (≈9–95% relative depth). Compared
to Claude Sonnet 4.5 (workspace ≈ L38–92 on a 0–100 axis), Qwen3.5-4B's sensory
block is proportionally much smaller (~first 9% vs ~first third).

See `results/qwen3.5-4b_cka_annotated.png`.

## Next steps

- Add the paper's auxiliary curves (next-token accuracy, J-lens excess kurtosis,
  top-concept autocorrelation, effective linear dimensionality) to trim the
  motor tail out of L3–L30.
- Rerun on `Qwen/Qwen3.6-27B` (same lens repo) — swap the constants in `cka.py`.

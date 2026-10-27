# UltraChat rotated16×8 failure analysis

**ARC 8/20 comes entirely from predicting A for every question.** The 20 answer keys contain 8 A, 7 B, 4 C and 1 D. The older handwritten rotated16×8 run also predicts A on all 20 and gets the same eight correct. This is a strong answer-letter preference on this subset, not evidence of retained reasoning across the questions.

| Saved result | ARC | Mean top probability within four choices | Mean probability mass of all four letters in full vocabulary |
|---|---:|---:|---:|
| Original BF16 | 19/20 | 0.927309 | 0.979091 |
| Historical handwritten16×8, packed | 8/20 | 0.819824 | 0.008823 |
| Current UltraChat 16×8, packed | 8/20 | 0.557875 | 0.024359 |

The first confidence column renormalizes only the four answer-letter scores; it is not a calibrated probability of correctness. Full-vocabulary letter mass is computed directly from saved one-token log-likelihoods. No EOS logits were saved for these ARC prompts, so the remaining mass cannot be attributed specifically to EOS.

Compared with BF16, 12 previously correct items become wrong, 7 remain correct, and the one BF16 error becomes correct because its answer is A. The gain is `Mercury_7006213`.

Lost baseline-correct IDs:

- `Mercury_SC_416516`
- `Mercury_SC_LBS10170`
- `Mercury_7040863`
- `Mercury_7064050`
- `VASoL_2009_5_25`
- `Mercury_7164658`
- `Mercury_7130498`
- `Mercury_7069003`
- `Mercury_183190`
- `Mercury_189018`
- `Mercury_7107363`
- `Mercury_SC_416156`

All current hard-FP32, same-GEMM FP16 and packed representations predict A on all 20. Their four generated answers are likewise identical: empty, empty, `Write a`, `#`, all followed by EOS. The older run generates `Here\n\n` followed by `This\n\n` 63 times, hitting 128 tokens without EOS on all four prompts. This is a behavioral change from repetition to premature termination; both are unusable on these tasks.

Current packed NLL is 8.498620, versus 9.255346 for the historical run (difference -0.756726); BF16 is 2.423777. Current hard-FP32 already has NLL 8.499410 and the same failed outputs before export. Exact fresh serialization is a separate passed check, not quality recovery.

## Hidden-state progression

These are corresponding layer outputs for one fixed NLL probe. Original teacher RMS values match across runs. Cosine and relative RMSE describe different aspects of drift; neither is a task score.

| Layer | Current cosine | Historical cosine | Current relative RMSE | Historical relative RMSE | Current / teacher RMS | Historical / teacher RMS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.984762 | 0.983048 | 0.177911 | 0.188618 | 0.972660 | 0.955919 |
| 7 | 0.818105 | 0.805806 | 0.577087 | 0.593515 | 0.774738 | 0.771977 |
| 15 | 0.739649 | 0.730238 | 0.689069 | 0.700109 | 0.886591 | 0.882250 |
| 23 | 0.536462 | 0.527622 | 0.900931 | 0.905545 | 0.851784 | 0.841226 |
| 31 | 0.411973 | 0.382131 | 1.093587 | 1.071043 | 1.016696 | 0.923604 |

Current cosine is higher in 32/32 layers and relative RMSE lower in 26/32. At the last layer, cosine improves but relative RMSE worsens; the student RMS is also larger. Similar RMS magnitude does not mean similar hidden vectors. This probe cannot identify a responsible family or establish a general distribution.

## Projection-family sampled weight statistics

All 200 projection names, sampled row IDs, original and rotated-before-correction moments match. The table gives per-family medians over the same sampled rows. Effective weights here are rotated original weights plus low-rank correction **before ternary quantization**, not decoded packed weights. Zero occupancy is separately counted across the full packed projections.

| Family | Count | Rotated-original kurtosis | Effective kurtosis, historical / current | Effective max/RMS, historical / current | Effective/rotated-original RMS, historical / current | Ternary zero fraction, historical / current |
|---|---:|---:|---:|---:|---:|---:|
| linear_attn.in_proj_qkv | 24 | 3.098487 | 3.098510 / 3.098478 | 4.396072 / 4.395394 | 1.000006 / 1.000006 | 0.309051 / 0.309051 |
| linear_attn.in_proj_z | 24 | 3.203244 | 3.203181 / 3.203264 | 4.723180 / 4.718311 | 1.000042 / 1.000037 | 0.309337 / 0.309294 |
| linear_attn.out_proj | 24 | 3.008817 | 3.008891 / 3.008777 | 4.493434 / 4.494030 | 1.000004 / 1.000009 | 0.309923 / 0.309735 |
| mlp.down_proj | 32 | 3.004223 | 3.004154 / 3.004187 | 4.446234 / 4.445636 | 1.000005 / 1.000003 | 0.310261 / 0.310047 |
| mlp.gate_proj | 32 | 3.118679 | 3.118480 / 3.119339 | 4.609020 / 4.612730 | 1.000210 / 1.000140 | 0.309573 / 0.309502 |
| mlp.up_proj | 32 | 3.051834 | 3.051567 / 3.051649 | 4.312604 / 4.315668 | 1.000092 / 1.000081 | 0.309695 / 0.309645 |
| self_attn.k_proj | 8 | 3.404184 | 3.404698 / 3.404280 | 4.927897 / 4.928956 | 0.999979 / 0.999990 | 0.309083 / 0.309051 |
| self_attn.o_proj | 8 | 3.013092 | 3.013417 / 3.013525 | 4.301404 / 4.303151 | 1.000009 / 1.000017 | 0.310578 / 0.310436 |
| self_attn.q_proj | 8 | 6.548717 | 6.545225 / 6.544199 | 7.007743 / 7.015324 | 1.000000 / 1.000000 | 0.309218 / 0.309216 |
| self_attn.v_proj | 8 | 3.150388 | 3.150658 / 3.150651 | 4.502263 / 4.501474 | 1.000004 / 1.000004 | 0.310265 / 0.310117 |

These moments and occupancies do not measure learned ternary-scale error, isolate a family, or explain the EOS behavior. Per-projection values and hashes are retained in the JSON.

## Calibration context and limits

| Calibration source | Raw EOS /16 | Raw caps /16 | Consumed truncations /16 | Truncated tokens | Active positions ×8 |
|---|---:|---:|---:|---:|---:|
| Historical handwritten | 13 | 3 | 3 | 96 | 41,872 |
| Current UltraChat nested16 | 3 | 13 | 13 | 927 | 59,856 |

The current selected teacher set has no empty/EOS-only continuation: the shortest contains 79 generated tokens. It contains fewer EOS completions and more capped/truncated traces than the historical teacher set, while the current compressed generations terminate earlier. This observation does not support a simple claim that frequent teacher EOS directly caused the current empty answers; it also does not rule out interactions involving sequence content or truncation. No causal intervention was performed.

The shared 16×8 optimizer recipe has the same update count but current active-token exposure is 1.429499× historical exposure. Dataset, token lengths and consumed contexts differ, so the earlier run is descriptive context. The authorized matched contrast is the newly frozen UltraChat 16×8 versus 128×1 pair.

No prompts, selections, hyperparameters or training were changed after examining these results. The 128×1 arm remains authorized independently of this result. This analysis does not recommend more training or claim a recovered model.

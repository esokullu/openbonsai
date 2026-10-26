# Compression method and reference boundaries

OpenBonsai is an independent adaptation of ScaleQ-style calibration to native dense Qwen3.5. The public papers and released CAT-Q inference code inform the method, but do not provide a complete training implementation that can be reproduced exactly. These notes distinguish observed reference behavior from OpenBonsai's training choices.

## What the public CAT-Q release establishes

The [BitTern CAT-Q release](https://github.com/IntelChina-AI/BitTern/tree/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q) was refreshed on October 26, 2026; `origin/main` remained at `5a8fcd4f7e0366554b732d300a37e6ea467c3c35`. Its [README](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/README.md#evaluation) describes checkpoints, inference, evaluation, and deployment, with training code to be released separately.

- The [low-rank wrapper](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/quantize/int_linear_lora.py#L11-L79) adds `B @ A * (lora_alpha / r)` to the original weight before the ternary quantizer. Its default `lora_alpha` is 1. The zero-filled, non-trainable `A` and `B` tensors in this loader are checkpoint placeholders; they do not establish the authors' training initialization.
- The [inference quantizer](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/quantize/quantizer.py#L95-L119) recomputes group statistics from the effective weight presented to it, then applies the learned factors. The [exporter](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/quantize/ternary_export.py#L101-L107) includes the low-rank update before producing ternary weights.
- The released [Qwen3-8B configuration](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/configs/qwen3-8b/config.yaml) records rank 64, low-rank learning rate `0.001`, and factor learning rate `0.005`. The [Qwen3-4B configuration](https://github.com/IntelChina-AI/BitTern/blob/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q/configs/qwen3-4b/config.yaml) instead records `0.0006` and `0.003`. There is no single model-independent release learning rate.
- The [CAT-Q paper](https://arxiv.org/html/2606.26650v1#S2.SS3) describes using gradients from the final soft iteration for subsequent hard-stage updates. The released inference code contains no training backward implementation, so it cannot resolve how that rule was implemented.

The paper's Appendix B lists factor learning rate `0.001`, and its reconstruction objective is an L2 error. The release configurations instead record model-specific factor rates and Huber loss. OpenBonsai defaults to MSE, factor rate `0.001`, and its original exponential factor mapping. Selectable sigmoid mappings and Huber loss support controlled comparisons with the release. The Huber threshold and full training implementation are not published in that release, so selecting these options does not establish an exact training reproduction.

## Original teacher and refined student

The original projection weight `W` remains frozen. With refinement rank `r > 0`, the student uses:

```text
A: [r, input_features]
B: [output_features, r]
W_effective = W + (B @ A) / r
```

OpenBonsai initializes `A` with Kaiming uniform (`a = sqrt(5)`) and `B` with zeros. The initial update is therefore zero while the low-rank branch can begin learning. This is an explicit training choice, not an initialization recovered from the public checkpoint loader.

For rank greater than zero, group means and centered mean absolute deviations are recomputed from `W_effective`, with derivatives flowing through those statistics. The learned group factors then control redistribution, reconstruction scale, and threshold. The original teacher ignores the low-rank update. Progress zero also returns the original dense weight; positive progress runs the student quantization path.

Window target provenance is an independent choice. By default, both original
and quantized windows receive the quantized prefix's input. The experimental
`--teacher-prefix original` control instead obtains the target from the original
prefix while preserving the student's quantized input. The [source audit and
paired control](teacher-target-control.md) explain the distinction; the papers'
shared-input notation does not establish the missing trainer's cache semantics.

The learning-rate and refinement defaults are:

```bash
python -m openbonsai train ... \
  --lora-rank 64 \
  --learning-rate 0.001 \
  --lora-learning-rate 0.001
```

`--learning-rate` controls the three group factors; `--lora-learning-rate` controls `A` and `B`. The optimizer applies the same linear decay schedule to both groups. Original model weights do not receive optimizer updates.

`--lora-rank 0` disables refinement and uses the original factor-only group statistics. This is an ablation of refinement; it still uses the corrected hard-stage derivative behavior. Rank is the number of low-rank channels and does not change the model's deployed architecture.

## Factor mappings and reconstruction loss

Both `train` and the offline `smoke` command accept these independent options:

```text
--factor-mapping exp|sigmoid       default: exp
--reconstruction-loss mse|huber   default: mse
--huber-delta POSITIVE_NUMBER     default: 1.0
```

For group mean `mu0`, initial scale `alpha0`, and learned raw parameters `u`, `v`, `w`:

| Quantity | `exp` (original behavior) | `sigmoid` (released CAT-Q mapping) |
|---|---|---|
| Initial scale | `max(mean(abs(W_effective - mu0)), 1e-8)` | `mean(abs(W_effective - mu0)) + 1e-6` |
| Adjusted mean | `mu0 + tanh(u) * alpha0` | `mu0 + (2 * sigmoid(u) - 1) * alpha0` |
| Reconstruction scale | `exp(v) * alpha0` | `2 * sigmoid(v) * alpha0` |
| Ternary threshold | `0.5 * exp(w)` | `0.5 * 2 * sigmoid(w)` |

The sigmoid option therefore changes both the factor mappings and the scale epsilon. Padding is excluded from statistics in both modes. Rank zero uses the selected mode's cached source-weight statistics; positive rank recomputes them from the corrected weight. The existing last-soft Jacobian mechanism differentiates the selected mapping, including the low-rank update. Export still stores only the resulting codes and scales, so both modes use the existing packed formats.

Huber uses `0.5 * error**2` when `abs(error) <= delta`, otherwise `delta * (abs(error) - 0.5 * delta)`. Errors are computed in FP32, padding is masked out, and the summed loss is divided by the number of nonpadding hidden-state elements across the accumulated batch. Delta must be finite and positive. The default `1.0` is an explicit OpenBonsai choice, **not** a recovered CAT-Q training setting; it is not automatically scheduled.

The existing MSE objective and reduction remain unchanged. Training losses, callbacks, and window records identify the selected objective and active Huber delta. `hard_mse_before` and `hard_mse_after` always report MSE, even during Huber training. Compare those diagnostic values across objectives rather than comparing raw Huber and MSE loss values as if they were the same metric. The mapping selection, which determines the scale guard, is recorded in training metadata.

To exercise the new path without model downloads:

```bash
python -m openbonsai smoke --output runs/tiny-sigmoid-huber \
  --factor-mapping sigmoid --reconstruction-loss huber --huber-delta 1.0
```

The same flags can be added to a fresh real-model `train` invocation. They do not change the learning rates, calibration budget, or projection policy automatically, and they cannot be applied to an already-packed artifact by repacking.

## Hard-stage derivative

The default `frozen_jacobian` mode records the actual last soft forward's factors, low-rank parameters, and sharpness. In the trainer this capture occurs **before that forward's optimizer update**. During hard-stage training:

1. Forward values come from hard ternarization of the current refined weights.
2. Backward uses the full soft-map Jacobian evaluated at that recorded soft point, including the low-rank branch and its group-statistic dependencies.
3. That Jacobian acts on the current upstream loss gradient; the previous batch's loss gradients are not reused.

This corrects the earlier use of post-update factors at an assumed terminal sharpness of 30. A discrete schedule may never have run a soft forward at exactly 30. The snapshot is refreshed by gradient-enabled soft forwards with trainable quantizer parameters; evaluation under `no_grad` does not replace it. Restarting or decreasing progress clears it, and the trainer releases it when a window finishes. The implementation stores parameter snapshots rather than a persistent dense effective-weight snapshot.

The schedule needs at least one soft training update before entering the hard stage. A one-update window cannot satisfy this and is rejected. Increasing the number of steps changes the early sharpness and the number of soft and hard updates; a tiny execution pilot is not evidence of convergence.

The optional `soft_surrogate` Python mode differentiates the soft map at the current parameters. It is an alternative experiment, not the CLI default. Neither interpretation is verified against the authors' unreleased training code.

## Export and verification

Export computes hard ternary codes and scales from the refined student. It saves those codes, FP16 scales/biases, and the retained floating-point tensors. It does not save a separate low-rank inference adapter. Packing remains BITCOS, legacy two-bit, or automatic selection; the low-rank change does not require a new packed format.

`openbonsai repack` only changes the encoding of already-ternary weights. It cannot add refinement, rerun calibration, or apply a training derivative fix. To test this update, start a fresh compression run from the original BF16 checkpoint and write a new output directory. Reusing valid traces from that same original checkpoint is possible; preserve their tokenizer and revision provenance.

Run a new memory preflight because low-rank parameters, optimizer state, snapshots, and effective-weight reconstruction add training costs. Peak memory measured for a factor-only pilot does not establish that the rank-64 path fits. The current reference runtime still reconstructs floating-point matrices for inference and supplies no optimized ternary kernel.

Verification must separate four questions: whether the original-weight teacher is unchanged; whether calibration improves hard-ternary behavior; what rounding FP16 export scales introduce; and whether saving/reloading the same packed representation is exact. Software tests and exact packed reload establish implementation properties, not recovered reasoning or language quality. Compare the new artifact and the original BF16 model on identical held-out inputs before making quality claims.

# RCO source audit

Audited 2026-10-26. Upstream [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO) is pinned to `9a1e09c07d468109cbe60a1b87d5036034a79d10` (June 13, 2026). License: Apache-2.0. Latest [paper](https://arxiv.org/abs/2605.00649) is v3, September 27, 2026. Preserve the upstream license, source attribution, pin, and descriptions of local modifications when redistributing code.

## What is actually reused

`src/manifold.py` supplies budget normal, tangent projection, adaptive binary-search retraction, and Adam first-moment transport. These are the RCO core, not a sensitivity allocator. `src/search/quant.py:578-780` supplies the algorithmic reference: Gumbel noise, hard budget-constrained DP choices with softmax straight-through derivatives of the SAME perturbed logits, joint model-loss backward, tangent projection, Adam update, retraction, and momentum transport. A decomposition into independent tensor sensitivities alone is a baseline and must not be labeled RCO.

For logits a and per-row candidate costs c, p=softmax(a); C=sum(p*c); n=p*(c-row_sum(p*c)). Projection removes dot(grad,n)/dot(n,n)*n. Retraction shifts a by t*c and bisects t to restore the expected budget. Transport projects Adam exp_avg onto the new tangent plane. Only exp_avg is transported. Use an optimizer containing alpha alone: upstream transport loops all optimizer parameters and assumes their shape matches alpha.

The unmodified primitives broadcast [N,K] tensor-specific costs despite documenting a shared [K] vector. Verified on CPU float64: normal matches autograd exactly, retraction reached 28.999999999903487 for target 29, projected normal inner product -1.76e-13. Normalize byte costs to avoid poor numerical scaling and explicitly validate the feasible budget interval, finite tensors, zero-normal cases and post-retraction residual. Upstream retraction does not reject infeasible targets.

## Ragged menus and discrete costs

The CLI and weight store require a shared integer bitwidth menu; bitwidth-map changes a shared cost per nominal bitwidth, not a tensor-specific codec/metadata cost. It cannot directly represent ternary + 4/6-bit + BF16 menus with different eligible options and zero-dependent BITCOS sizes.

Use explicit per-tensor candidates with actual serialized/payload costs. Ragged rows require validity masks; never materialize missing options or let them receive probabilities. Fixed tensors and file overhead belong in a fixed budget offset. Dtype, scale/zero bytes, group padding, bias, packing choice and decoder must be consistent with the candidate evaluated during search.

Do NOT copy the upstream DP implementation without adapting it:

- `budget_constrained_argmax` rounds costs to a grid (default 500). Reproduced an assignment costing 2.80024 at target 2.8. This is consistent with the paper's rounding caveat but not an exact byte-budget guarantee.
- `round_with_budget_dp` (default resolution 10000) chooses the largest reachable budget, rather than the highest summed log probability over all feasible budgets. Reproduced one-row p(1)=.9/p(2)=.1, budget 2: it selects 2. The forward sampler uses max score over all feasible costs, so the two decoders differ.
- Both need explicit infeasibility/backtracking checks; the sampler uses a finite negative sentinel and does not check every predecessor is reachable.

Use a tested multiple-choice knapsack decoder with integer byte costs and max score over ALL feasible totals. Subtract each row's minimum cost and divide cost deltas by their GCD when possible. If a dense exact DP is too large, conservative rounded-up delta costs with rounded-down remaining budget guarantee final byte feasibility but may exclude valid assignments; label this approximation and still assert actual total bytes. RCO's exact continuous expectation does not prove optimal discrete quality.

## Candidates: GPTQ, ScaleQ and BITCOS

The repo ships GPTQ-derived Cholesky OBQ candidate generation (`src/quant/fast_obq.py`), ordinary uniform affine integer quantizers, a multi-bitwidth database, and real compressed-tensors export. There is no GSQ implementation or GSQ integration in this pin or paper. An integer cost mapping of 2 to 1.58 does not implement ternary quantization. OpenBonsai's trained ScaleQ-inspired ternary checkpoint can be a separate candidate; higher-precision candidates must come from the unchanged original BF16 source, never from the ternary approximation. BITCOS is lossless encoding of those fixed ternary codes; it does not improve their quality.

The published search assembles weights from a highest-bit reference plus dequantized BF16 residuals. This is not a packed inference kernel and can consume substantial memory. Use bounded materialization, checkpointing, and CPU candidate storage on a 5090, with exact candidate-forward/reload parity checks. Stock export's fake-quant mode saves BF16 and gives no disk savings; compressed-tensors mode packs supported uniform quantization schemes, not arbitrary BITCOS mixtures.

## GPTQ adaptation details

FastOBQ assumes input width divisible by group_size. Its group_size=None path leaves returned local scale/zero arrays unfilled; explicitly require positive divisible groups. Block size 128 with group size 128 aligns parameter estimation with the error-feedback blocks. Non-divisible group/block cases should fail before modifying any source tensors.

The quantization loop computes FP32 scales but copies them into arrays using original weight dtype. A BF16 original can therefore produce quantization decisions from FP32 scales and export BF16 rounded scales. Wrapping detached original rows as FP32 linear weights preserves FP32 qparams; use these same persisted qparams for search and export. Any later FP16 scale conversion is a separate lossy step that needs measurement.

Asymmetric zero points are rounded integers and can be negative for strictly positive groups. Preserve them rather than assuming they are unsigned packed codes. Symmetric zero is the midpoint 2^(bits-1), NOT zero despite stale qparams comments. Disable act_order initially; otherwise preserve and invert the exact column permutation when materializing candidates. Preserve original weights/Hessian: FastOBQ mutates its H and private working W, so clone H for each independent row chunk.

Embedding lookup does not expose the linear-layer input Hessian assumed by GPTQ. A grouped per-row RTN candidate using the upstream affine Quantizer and original BF16 embedding is a useful explicit baseline. An identity-H FastOBQ call reduces to this diagonal-metric case; do not describe it as activation-calibrated GPTQ. Head and decoder projections can use the original model's real hidden-state input covariance. Capture on original-model forwards with bounded hooks and exclude padding; label the calibration source and token counts.

## Evaluation and practical limitations

Use untouched held-out evaluation separately from allocation/calibration loss and experiment-selection validation. RCO optimizes a joint CE or teacher KL loss; teacher KL upstream caches the full vocabulary distribution, which is expensive at Qwen's 248320 vocabulary. Stream or cache bounded teacher batches and label top-k approximations if used. Keep unchanged BF16 baselines, fixed source revisions, candidate hashes, actual byte accounting, and mixed export/reload parity.

Upstream CLI defaults differ from the updated paper recommendations. Paper v3 recommends 16-32 Gumbel samples, 50-200 steps, tau_min=.01, LR=.1; a smaller test is an engineering pilot, not paper reproduction. The implementation's actual temperature schedule is exponential even though a docstring says cosine. Qwen3.5 hybrid attention and the ScaleQ+RCO+BITCOS composition are new experiments, not validated results from the RCO paper.

Parent authentication passed: all 2,034 sealed files / 16,192,227,790 bytes match seal `d7ae0ccf66b38f223366beb105beba32d651e5c2c1be166c21624eac7885bf8a`. No parent file was modified or parent package imported.

The rotated128×1 checkpoint contains exactly200 module dictionaries, each with raw_mu, raw_scale, raw_threshold, lora_A and lora_B in FP32 plus signed int8 rotation_signs. All1,000 parameter payload hashes match the independent saved-state audit; all200 sign arrays match initialization and the independent deterministic sign recipe.

Both original and learned-hard probe files bind to the same exact31 token IDs, mask, diagnostic input hash and full-vocabulary final-position FP32 logits. These are authenticated saved references; no fresh model/probe execution has occurred.

Required reconstruction steps:

- Use an authenticated new-run source copy; never import or edit the sealed checkout.
- Load the exact pinned official native model in BF16 with eager attention and no remote code; authenticate original source before load.
- Construct the same200 rotated decoder wrappers with group128/rank64/sigmoid/block4096/seed5090; original frozen weight fingerprints must match.
- Preserve native retained endpoints/vision/recurrent tensors; regenerate mu0/alpha0 from the original weights using the frozen rotated constructor and verify against initial fingerprints.
- Copy all five learned tensors per module exactly in FP32 and all signed int8 rotation arrays; do not cast learned state to BF16 or FP16.
- Set eval mode, enable hard quantization at progress1, and use the inherited original-weight teacher path; no training, optimizer, soft-snapshot reuse, scale rounding, packed artifact or decoded-cache substitution.
- Before activation diagnosis, reproduce the original and hard learned fixed probes with the exact31 IDs/mask, eager/noKV/logits_to_keep1; require bitwise equality of stored FP32-cast logits.

Runtime gates remain pending:

- new source copy/instrumentation authentication
- fresh pinned model source authentication/load
- restored state/derived-statistics byte identity
- original BF16 fixed-probe equality
- restored learned-hard fixed-probe equality

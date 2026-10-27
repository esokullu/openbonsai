# Runtime guard failure: CPU diagnosis

The guard compared a native Python configuration dictionary directly with JSON readback. The CPU reproduction identifies exactly three `id2label` dictionaries whose keys change representation: `original_config.id2label`, `original_config.text_config.id2label`, and `original_config.vision_config.id2label`. Each native dictionary has integer keys `0` and `1`; JSON readback has string keys `"0"` and `"1"`. Values remain `LABEL_0` and `LABEL_1`.

The two saved runtime receipts are exactly equal as parsed JSON and their raw bytes are also identical. Current SHA256: `d4b711dde86c0fbcf364f0f641929717fb51a82cf9b686a2973156d41fbc037c`; v2 SHA256: `d4b711dde86c0fbcf364f0f641929717fb51a82cf9b686a2973156d41fbc037c`.

CPU `AutoConfig.from_pretrained` loaded only the pinned official configuration, without model construction or weights. The installed native parser explicitly converts `id2label` keys to integers. Reconstructing that native configuration from the exact saved runtime configuration gives direct inequality at those three locations and no other differences. Its JSON roundtrip exactly equals the saved configuration. Substituting this reconstructed configuration into the recorded runtime metadata reproduces the failed direct comparison and successful JSON-equivalent comparison.

The failed model process took 24.906000 seconds; the completed original BF16 load took 15.280383 seconds, peaking at 18,831,105,024 allocated and 18,884,853,760 reserved bytes. The assertion ran before original-model probes, research-state restoration, paired caching, the four discarded updates, either repair arm, quality evaluation, delta export or fresh verification. Optimizer updates: **0**.

**Limit:** the live pre-serialization dictionary and typed mismatch were not saved by the failed process. This is an exact CPU reproduction using the installed native config parser and pinned file, not recovery of that live object. Saved metadata equivalence does not establish model-output parity; those gates did not run.

No frozen source or prior-run file was changed, no model was instantiated by this audit, CUDA remained uninitialized, and no GPU retry or comparison fix was applied. The JSON receipt binds all observed files and the CPU helper by SHA256.

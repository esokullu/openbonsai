Initial independent review of sealed v1 measurement path

Read all four source files below in full before reviewing the proposed v2 changes. No additional blocking scientific defect was found. This is a source review, not execution of the 9B diagnostic. No GPU, model load, training, or source edits were performed.

| Sealed v1 source | SHA256 |
|---|---|
| tools/activation_measurement.py | befbbcfce15c3c9c96896ce1673b2312b81c5fe32e190985fb510aedcfe8f255 |
| tools/restore_learned_state.py | 3dc1a7b1b75cb2dc13f86789f7d28f92f6896a4aeae9951862c469e5998163ac |
| tools/measure_activation.py | 52584bded6002807647e51294b16f66342239d8731e1fc35e754128ca5c2ef09 |
| checkout/openbonsai/activation_capture.py | 00dc7f35fb0bdbaff04fb2a52207a8cd5ba68a0faf79958294dcfe5da2f64279 |

The preserved scientific path requires exactly 16 records, batch one, length512 and hidden width4096 at windows [0,1) and [16,17). First-two counted captures, independent original/conditional/hard native paths, A-B-A/reverse equality, first-window zero target gap, active-token element counts, finite padding, signed error decomposition and the predeclared 1e-12 relative closure gate are coherent. The 16-denominator selector treats zero/near-zero denominators as indeterminate and is neither a quality gate nor training authorization. See activation_measurement.py:182,280,311,344 and activation_capture.py:118.

The meter retains bounded free-memory minima and allocated/reserved peaks; sampled and conservative free headroom must both be at least4 GiB. The preflight uses charged elapsed +1.25 times remaining capture cost +300 seconds, then checks the actual absolute deadline as well as the original2700-second envelope (activation_measurement.py:92,351,442). Thus an earlier v2 deadline can enforce the carryover debit without any change to this scientific helper.

Restoration verifies original BF16 source storage, ordered coverage of200 modules, all saved FP32 factor/A/B tensors and explicit signs, recreated initial fingerprints/RNG, hard progress1, frozen parameters and absent snapshots. No optimizer or last-soft snapshot is recreated. Exact original/bypass/hard probe parity remains required in the caller (restore_learned_state.py:88,184,234).

The observed v1 failure is the raw-file-versus-loaded-template hash assertion at measure_activation.py:173–174. It follows original BF16 loading and precedes original tensor hashing, learned-state restoration and activation collection. The v1 supervisor charges23.503059148788452 seconds, leaving2676.4969408512115 seconds of the original2700; a2676-second v2 cap is conservative.

Three orchestration assumptions require explicit authenticated v2 handling: the template file/runtime representation guard; producer source-freeze identity versus the separate consumer source freeze; and restore receipt inventory.path, whose v1 check requires a path inside its current RUN (restore_learned_state.py:96–98). A strict adapter should bind the exact sealed v1 inventory without rewriting its receipt. The v1 driver supervisor equalities at lines114–115 must also accept the explicitly debited v2 deadline.

No scientific helper change is proposed or authorized by this review. The upcoming review will compare the full v2 driver, metadata/path adapters, template guard and tests, and confirm byte identity of measurement/collector/statistics and inherited quantizer/native core.

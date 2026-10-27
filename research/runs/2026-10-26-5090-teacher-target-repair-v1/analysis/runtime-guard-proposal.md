# Unapplied runtime guard proposal

The standalone [patch](proposed_runtime_guard.patch) changes only the equality guard to compare the just-persisted current runtime JSON against the saved v2 runtime JSON. Both operands then have the same JSON representation; attention, precision, source hashes and all configuration values remain subject to strict equality. The original live object remains available for unchanged later in-process state comparisons.

Seven CPU tests passed in 5.11 seconds: the exact pinned native configuration reproduces and accepts only the integer/string-key serialization distinction; changes to attention, TF32, native source hashes, hidden size or label values are rejected. The tests compile only the proposed assertion, never import or execute the model driver, and verify the frozen driver hash is unchanged. CUDA remained uninitialized.

**Not applied.** No frozen source was edited, no model retry ran, and the full driver with this proposal is unexecuted. This failed attempt remains stopped. The [validation receipt](runtime-guard-proposal-validation.json) binds the proposal, tests and original frozen source.

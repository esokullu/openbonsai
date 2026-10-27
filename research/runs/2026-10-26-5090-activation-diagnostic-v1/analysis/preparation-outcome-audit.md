# Preparation outcome audit

Passed: 16 distinct search-validation records accepted in 16 attempts from the exact frozen unused-reserve order. Excluded attempts: 0. These records select an action; they are not an unused final confirmation set.

| Quantity | Measured |
|---|---:|
| prompt tokens | 895 |
| generated tokens | 7,023 |
| raw tokens | 7,918 |
| active positions | 7,203 |
| padding tokens | 989 |
| consumed generated tokens | 6,308 |
| eos count | 4 |
| cap count | 12 |
| truncated records | 12 |
| truncated tokens | 715 |
| eos retained by consumption | 4 |
| eos removed by consumption | 0 |
| eos only continuations | 0 |

The supervisor completed successfully in 299.839 s of 1200 s. Loading took 17.496 s; generation phases took 274.447 s (25.590 generated tokens/s). The total cap includes interpreter startup, source authentication, model loading, generation, final validation and freezing.

Every consumed record independently matches the raw trace’s first 512 IDs, followed by pad 248044; masks are exactly 1 on active positions and 0 on padding. Raw-row hashes, source prefixes, all 16 trace files, ordered attempt IDs, file byte counts, settings and source freezes match. EOS-only and capped traces are valid under the original policy; no response usefulness or semantic content was used to choose or reject a record.

All 108 runtime source files still match the frozen hashes. The settings bind the preserved running supervisor receipt; the final receipt changes only its completion status/metadata while retaining the same command, process ID, environment and budget.

No exact prompt-ID, whitespace-normalized prompt, full prompt-token, complete raw-token, or complete consumed-token matches exist against the 144 prior fitting prompts. The pre-generation source freeze additionally checked all enumerated earlier search input registries and preflight aliases. These exact checks do not exclude semantic topic overlap or shared individual tokens.

Raw traces: `9c1c05142d948d8cb88e4770c3ac2695afb91b1f8534802ba4d81dd780b80d86`; consumed records: `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf`. See preparation-outcome-audit.json for all hashes, per-record counts and exclusion evidence.

# Historical preservation verification

Result: **PASS**. All checks used read-only, single-threaded SHA256 streaming with an 8-MiB read buffer. No GPU runtime was imported, no old run was rewritten, and the current checkout was untouched.

| Authenticated inventory | Listed files verified | Inventory bytes |
|---|---:|---:|
| Stopped UltraChat parent | 155/155 | 283,880,925 |
| Prior rotations run | 2465/2465 | 15,946,218,435 |
| Historical unique files referenced by rotations | 1379/1379 | 28,151,646,035 |

Total: **3,999 unique inventory files / 44,381,745,395 bytes** rehashed, plus verification of the authenticated seal/inventory anchors. Elapsed: 45.595 seconds.

Stopped-parent seal: `5bd642801cf68c5250e5c56710929ac6c4da1a48e6e827da31607f509f2319d6`.
Prior-rotation seal: `d723275e9057ed27c8bb7e2b010c0e28ef5fa517ffb95294cac76830ccbb6c2f`.
Historical inventory SHA256: `61cc74a441579588f47e50e5054e90635bb20caa4ed60822c581c14f810bb7e9`, authenticated by the prior-rotation seal. All anchors were unchanged at completion.

The 1379-file historical scope contains:

- `D:\openbonsai\runs\2026-10-26-5090-pilot`: 232 listed files, 6,852,166,415 bytes.
- `D:\openbonsai\runs\2026-10-26-5090-diagnosis`: 675 listed files, 13,416,405,972 bytes.
- `D:/openbonsai original local source files explicitly listed`: 30 listed files, 345,232 bytes.
- `D:\openbonsai\runs\2026-10-26-5090-corrected`: 442 listed files, 7,882,728,416 bytes.

The historical inventory includes 291 paths containing `bitcos`, totaling 6,582,272,426 bytes. This includes the listed packed-BITCOS payloads and supporting sources/reports, and is already part of the 1379-file total.

The older audit reported 1412 manifest references deduplicated to 1379 files. This current verification rehashed those 1379 unique listed files once.

Only exact listed paths, their recorded sizes/content hashes, and authenticated inventory/seal files were verified. Unlisted/new/deleted-outside-inventory paths, cache directories, other historical artifacts and unrelated branches are not covered.

Detailed per-file expected/actual hashes, sizes, and stability checks are preserved in the companion JSON. This verification ran while the separately authorized 128-sample experiment was active; it does not inspect or modify that experiment.

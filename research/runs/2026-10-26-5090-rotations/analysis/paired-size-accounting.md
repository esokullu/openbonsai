Both completed paired artifacts passed the final CPU audit. The unrotated deployable directory is **6,592,437,649 bytes**; the rotated directory is **6,596,776,129 bytes**, an increase of **4,338,480 bytes**. Each total already includes its tokenizer. These are storage measurements; they do not establish model quality or low-bit inference speed. [Complete audit](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/paired-artifact-audit.json)

Per condition, the audit inventoried **770 files**, verified all **769 manifest-listed file SHA256 values**, checked **200 candidate internal checksums**, and matched **560 persistent retained tensor-value hashes** to prior verified original/cast values. It checked all 564 retained descriptors, including four generated buffers. Both completion markers, independent fresh-process logit parity, and cached-versus-uncached exact-logit records passed. The same 200 original projection fingerprints and shapes are represented in both conditions. This audit performed no model inference.

Both conditions selected BITCOS for all 200 projections. Candidate payloads represent **6,912,212,992 original weights**; the two kinds of signs below have distinct purposes.

| Candidate tensor component | Unrotated bytes | Rotated bytes |
|---|---:|---:|
| Legacy 2-bit code stream | 0 | 0 |
| BITCOS presence bits | 864,026,624 | 864,026,624 |
| BITCOS nonzero-code sign bits | 593,249,076 | 596,473,766 |
| FP16 group scales | 108,003,328 | 108,003,328 |
| Projection biases | 0 | 0 |
| Explicit Hadamard rotation signs, int8 | 0 | 1,081,344 |
| BITCOS row-sign offsets/indexes | 11,404,864 | 11,404,864 |
| **Candidate tensor payload** | **1,576,683,892** | **1,580,989,926** |
| Candidate prefixes, JSON headers and checksums | 156,080 | 188,528 |
| **Complete candidate files** | **1,576,839,972** | **1,581,178,454** |

The rotated artifact's increase comprises **3,224,690 additional BITCOS sign-stream bytes**, **1,081,344 explicit rotation-sign bytes**, and **32,446 net metadata/container bytes**. Code streams alone use 1.686610 versus 1.690342 bits per targeted weight; all candidate tensors, including scales/signs/indexes, use **1.824809 versus 1.829793 bits per weight**.

Retained components are identical in both conditions:

| Retained tensor component | Bytes per condition |
|---|---:|
| Output head | 2,034,237,440 |
| Token embeddings | 2,034,237,440 |
| Vision encoder, including vision rotary buffers | 912,021,104 |
| Recurrent gates | 12,582,912 |
| Recurrent convolution | 1,572,864 |
| Text normalization | 546,816 |
| Recurrent dynamics | 3,072 |
| Text rotary buffers | 256 |
| **Retained tensor payload** | **4,995,201,904** |
| Retained safetensors headers | 66,216 |
| **Complete retained files** | **4,995,268,120** |

The complete directory partition is:

| File scope | Unrotated bytes | Rotated bytes |
|---|---:|---:|
| Candidate files | 1,576,839,972 | 1,581,178,454 |
| Retained files | 4,995,268,120 | 4,995,268,120 |
| Model configuration | 3,951 | 3,951 |
| Generation configuration | 2,059 | 2,059 |
| Manifest | 324,786 | 324,784 |
| **Model artifact excluding tokenizer** | **6,572,438,888** | **6,576,777,368** |
| Tokenizer assets | 19,998,761 | 19,998,761 |
| **Total deployable directory** | **6,592,437,649** | **6,596,776,129** |

Strict equivalent-source comparisons use **all candidate tensors plus persistent retained tensors**, including rotation signs and indexes, but excluding **400 generated rotary-buffer bytes**. The resulting numerators are **6,571,885,396** and **6,576,191,430 bytes**.

| Original denominator and scope | Denominator bytes | Unrotated reduction / factor | Rotated reduction / factor |
|---|---:|---:|---:|
| Same represented raw checkpoint tensors | 18,819,635,168 | 65.079634% / 2.863658× | 65.056754% / 2.861783× |
| Same persistent tensors after BF16 loading | 18,819,627,488 | 65.079620% / 2.863657× | 65.056740% / 2.861782× |

The original checkpoint contains **19,306,216,416 tensor bytes**. Its **15 unused MTP tensors, totaling 486,581,248 bytes**, are excluded from both denominators. BF16 loading converts **48 retained FP32 tensors** from 15,360 to 7,680 bytes, explaining the **7,680-byte denominator difference**. Four constructor-generated nonpersistent rotary buffers add 400 exported bytes—256 text and 144 vision—but have no original checkpoint tensor counterpart. All persistent retained values were verified as either original bytes or the previously verified BF16 casts. [Original scope/cast evidence](D:/openbonsai/runs/2026-10-26-5090-corrected/corrected16x1/artifact-accounting-audit.json)

An auxiliary complete-directory comparison adds the identical saved tokenizer/config allowance to the loaded persistent denominator: **18,839,632,259 bytes**. Reductions are **65.007610% / 2.857764×** unrotated and **64.984581% / 2.855885×** rotated. This is explicitly an asset-allowance comparison, not the size of a measured original Hugging Face directory; original tensor-container overhead is excluded.

The research checkpoint and decoded runtime cache are outside deployable disk totals:

| Excluded item | Unrotated bytes | Rotated bytes |
|---|---:|---:|
| Saved FP32 research-state file | 1,290,046,185 | 1,291,183,353 |
| Decoded BF16 projection cache in runtime memory | 13,824,425,984 | 13,824,425,984 |

The cache plus packed/retained tensor storage totals **20,396,311,780** and **20,400,617,814 bytes**, before activations, workspaces, allocator and driver overhead. The cache retains the same 64-row GEMM calls and is absent from artifact tensors. Research-file sizes were checked against saved metadata; research tensor contents were not decoded or revalidated by this artifact audit. Logs, reference logits, and duplicate staging tokenizer assets are also excluded.

Reproduction: run `python audit_paired_artifacts.py --verify-payload-hashes --output <new-output-path>` from the rotations run directory. The [auditor](D:/openbonsai/runs/2026-10-26-5090-rotations/audit_paired_artifacts.py) writes outside the frozen checkout, does not overwrite existing outputs, and leaves prior runs unchanged.

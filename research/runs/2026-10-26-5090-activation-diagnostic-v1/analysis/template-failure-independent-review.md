# Independent template failure review

The failed guard compared two byte representations of the same tested template. The saved parent file contains CRLF line endings; the official cached file and both tokenizer runtime strings contain LF. The parent file still matches its immutable manifest.

| Representation | Bytes | SHA256 |
|---|---:|---|
| Parent serialized file (153 CRLF lines) | 7909 | `a92e1dd97cb1cb175c9b70c0828e146bea4371c2643319b661b777e89811972e` |
| Official cached file / tokenizer.chat_template / get_chat_template() | 7756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| Parent file after CRLF→LF only | 7756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

Independent CPU checks found exact token-ID equality for all 16 frozen search prompts and the one allowlisted saved probe, using the runtime template, the explicit raw CRLF template, and the LF template. All 16 consumed prompt prefixes also match; rendered prompt bytes agree on all 17 cases. No responses or scores influenced these checks.

The GPU attempt stopped after loading the original BF16 model and before wrapper restoration, model probes, or activation capture. Those parity checks were not run. This CPU result identifies the serialization mismatch in the guard; it does not demonstrate model or activation parity. No model weights or saved logits were loaded for this review, CUDA remained uninitialized, and no source was changed or retry launched.

A future guard should authenticate the raw saved file against its immutable manifest, then compare runtime template text using an explicitly consistent representation and validate frozen prompt IDs separately. Newline normalization must not replace raw-file integrity authentication.

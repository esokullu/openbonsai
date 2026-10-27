// CPU-only audit ABI around Prism's existing PTQ1_0 reference codec.
// Build instructions and the required source pin are documented in
// docs/bonsai-ternary-statistics.md. No model context or inference is created.
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdint>
extern "C" void audit_init() {
    ggml_init_params p = {16384, nullptr, true};
    ggml_context * ctx = ggml_init(p);
    ggml_free(ctx);
}
extern "C" void audit_decode(const void * data, float * out, int64_t elements) {
    dequantize_row_ptq1_0(static_cast<const block_ptq1_0 *>(data), out, elements);
}
extern "C" void audit_encode(const float * data, void * out, int64_t elements) {
    quantize_row_ptq1_0_ref(data, static_cast<block_ptq1_0 *>(out), elements);
}

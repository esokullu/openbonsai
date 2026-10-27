// Thin reference evaluator for the pinned Prism llama.cpp C API.
// No chat rendering, implicit specials, samplers, truncation, or retained record state.
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OPENBONSAI_PRISM_SHA
#error "Build with tools/build_prism_eval.py to verify the pinned source"
#endif
#ifndef OPENBONSAI_CUDA_BUILD
#define OPENBONSAI_CUDA_BUILD 0
#endif

namespace prism_eval {
using json = nlohmann::json;
constexpr int kBatch = 64;
constexpr size_t kMaxLine = 16 * 1024 * 1024;
constexpr const char * kProtocol = "openbonsai.frozen-evaluation.v1";
constexpr const char * kReduction = "fp32-logits-fp64-logsumexp-v1";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int64_t integer(const json & value, int64_t low, int64_t high, const char * name) {
    require(value.is_number_integer(), std::string(name) + " must be an integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        require(number <= static_cast<uint64_t>(high), std::string(name) + " out of range");
        require(number >= static_cast<uint64_t>(std::max<int64_t>(low, 0)), std::string(name) + " out of range");
        return static_cast<int64_t>(number);
    }
    const auto number = value.get<int64_t>();
    require(number >= low && number <= high, std::string(name) + " out of range");
    return number;
}

json parse_request(const std::string & line) {
    // nlohmann otherwise silently keeps the last duplicate key.
    std::vector<std::set<std::string>> keys;
    auto callback = [&keys](int, json::parse_event_t event, json & parsed) {
        if (event == json::parse_event_t::object_start) keys.emplace_back();
        if (event == json::parse_event_t::key) {
            require(!keys.empty() && keys.back().insert(parsed.get<std::string>()).second,
                    "duplicate JSON object key");
        }
        if (event == json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    json result = json::parse(line, callback);
    require(result.is_object(), "request must be an object");
    require(result.contains("id") &&
                (result.at("id").is_string() || result.at("id").is_number_integer()),
            "id must be a string or integer");
    require(result.contains("op") && result.at("op").is_string(), "op must be a string");
    return result;
}

void fields(const json & request, const std::set<std::string> & expected) {
    require(request.size() == expected.size(), "missing or unexpected request fields");
    for (const auto & entry : request.items()) {
        require(expected.count(entry.key()) != 0, "unexpected field: " + entry.key());
    }
}

std::vector<llama_token> tokens(const json & values, int vocab_size, bool allow_empty = false) {
    require(values.is_array(), "token IDs must be an array");
    require(allow_empty || !values.empty(), "token IDs must not be empty");
    std::vector<llama_token> result;
    result.reserve(values.size());
    for (const auto & value : values) result.push_back(static_cast<llama_token>(integer(value, 0, vocab_size - 1, "token ID")));
    return result;
}

std::vector<int> positions(const json & values, size_t input_length) {
    require(values.is_array() && !values.empty(), "score_positions must be a nonempty array");
    require(input_length >= 2, "scoring requires at least two input IDs");
    std::vector<int> result;
    int previous = 0;
    for (const auto & value : values) {
        const auto position = static_cast<int>(integer(value, 1, static_cast<int64_t>(input_length) - 1, "score position"));
        require(position > previous, "score_positions must be strictly increasing and unique");
        result.push_back(position);
        previous = position;
    }
    return result;
}

double target_logprob(const float * logits, int size, int target) {
    require(logits != nullptr && size > 0 && target >= 0 && target < size, "invalid logit row");
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < size; ++i) {
        require(std::isfinite(logits[i]), "nonfinite logit");
        maximum = std::max(maximum, static_cast<double>(logits[i]));
    }
    // Center before subtracting the normalizer: stable even for huge finite logits.
    double normalizer = 0;
    for (int i = 0; i < size; ++i) normalizer += std::exp(static_cast<double>(logits[i]) - maximum);
    const double result = (static_cast<double>(logits[target]) - maximum) - std::log(normalizer);
    require(std::isfinite(result) && result <= 0, "invalid normalized target log probability");
    return result;
}

int argmax(const float * logits, int size) {
    require(logits != nullptr && size > 0, "invalid logit row");
    int best = 0;
    for (int i = 0; i < size; ++i) {
        require(std::isfinite(logits[i]), "nonfinite logit");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

std::string hex(const std::string & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 15]);
    }
    return output;
}

struct Options {
    std::string model;
    int context_length = 0;
    int threads = 0;
    int gpu_layers = 0;
};

int positive_cli_integer(const std::string & value, int low, const char * name) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            std::string(name) + " must be a decimal integer");
    return static_cast<int>(integer(json::parse(value), low, std::numeric_limits<int>::max(), name));
}

Options options(int argc, char ** argv) {
    Options result;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        require(i + 1 < argc, "every CLI flag requires a value");
        require(seen.insert(key).second, "duplicate CLI flag: " + key);
        const std::string value = argv[i + 1];
        if (key == "--model") result.model = value;
        else if (key == "--context-length") result.context_length = positive_cli_integer(value, 2, "context length");
        else if (key == "--threads") result.threads = positive_cli_integer(value, 1, "threads");
        else if (key == "--gpu-layers") result.gpu_layers = positive_cli_integer(value, 0, "GPU layers");
        else throw std::runtime_error("unknown CLI flag: " + key);
    }
    require(!result.model.empty() && result.context_length >= 2 && result.threads >= 1,
            "usage: prism-eval-worker --model PATH --context-length N --threads N [--gpu-layers N]");
    require(OPENBONSAI_CUDA_BUILD || result.gpu_layers == 0, "CPU build requires --gpu-layers 0");
    return result;
}

struct Batch {
    llama_batch value = llama_batch_init(kBatch, 0, 1);
    ~Batch() { llama_batch_free(value); }
    Batch() = default;
    Batch(const Batch &) = delete;
};

class Worker {
    Options options_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    const llama_vocab * vocab_ = nullptr;
    int vocab_size_ = 0;

    void clear() {
        auto memory = llama_get_memory(context_.get());
        require(memory != nullptr, "model has no resettable memory");
        llama_memory_clear(memory, true);
    }

    void check_length(size_t length) const {
        // Enforce requested cap even if llama rounds the allocation upward.
        require(length <= static_cast<size_t>(options_.context_length), "context overflow");
    }

    void decode(Batch & batch, const std::vector<llama_token> & ids, int offset, int count,
                const std::set<int> & output_sources) {
        require(count > 0 && count <= kBatch, "invalid decode batch size");
        batch.value.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.value.token[i] = ids.at(offset + i);
            batch.value.pos[i] = offset + i;
            batch.value.n_seq_id[i] = 1;
            batch.value.seq_id[i][0] = 0;
            batch.value.logits[i] = output_sources.count(offset + i) ? 1 : 0;
        }
        const int status = llama_decode(context_.get(), batch.value);
        require(status == 0, "llama_decode failed: " + std::to_string(status));
    }

public:
    explicit Worker(Options value) : options_(std::move(value)) {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = options_.gpu_layers;
        model_params.load_mtp = false;
        model_.reset(llama_model_load_from_file(options_.model.c_str(), model_params));
        require(model_ != nullptr, "model load failed");
        require(!llama_model_has_encoder(model_.get()), "only decoder-only models are supported");
        vocab_ = llama_model_get_vocab(model_.get());
        vocab_size_ = llama_vocab_n_tokens(vocab_);
        require(vocab_size_ > 0, "empty vocabulary");
        auto params = llama_context_default_params();
        params.n_ctx = options_.context_length;
        params.n_batch = kBatch;
        params.n_ubatch = kBatch;
        params.n_seq_max = 1;
        params.n_rs_seq = 0;
        params.n_outputs_max = kBatch;
        params.n_outputs_max_per_seq = kBatch;
        params.n_threads = options_.threads;
        params.n_threads_batch = options_.threads;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = GGML_TYPE_F32;
        params.type_v = GGML_TYPE_F32;
        params.offload_kqv = options_.gpu_layers > 0;
        params.op_offload = options_.gpu_layers > 0;
        params.no_perf = true;
        context_.reset(llama_init_from_model(model_.get(), params));
        require(context_ != nullptr, "context creation failed");
        require(llama_n_ctx(context_.get()) >= static_cast<uint32_t>(options_.context_length), "allocated context is smaller than requested");
        clear();
    }

    json metadata() const {
        std::vector<int> eog;
        for (int id = 0; id < vocab_size_; ++id) if (llama_vocab_is_eog(vocab_, id)) eog.push_back(id);
        return {{"protocol", kProtocol}, {"runtime_sha", OPENBONSAI_PRISM_SHA},
                {"vocab_size", vocab_size_}, {"context_length", options_.context_length},
                {"allocated_context_length", llama_n_ctx(context_.get())},
                {"cache_mode", "full-prefix-reference"}, {"reset_policy", "per-record"},
                {"reduction", kReduction}, {"logits_dtype", "float32"},
                {"model_path", options_.model}, {"threads", options_.threads},
                {"gpu_layers", options_.gpu_layers}, {"cuda_build", bool(OPENBONSAI_CUDA_BUILD)},
                {"batch_size", llama_n_batch(context_.get())}, {"ubatch_size", llama_n_ubatch(context_.get())}, {"kv_cache_dtype", "float32"},
                {"flash_attention", false}, {"load_mtp", false}, {"native_eog_ids", eog}};
    }

    json dispatch(const json & request) {
        // Reset also on tokenization/metadata requests, preventing retained record state.
        clear();
        const auto op = request.at("op").get<std::string>();
        if (op == "metadata") {
            fields(request, {"id", "op"});
            return metadata();
        }
        if (op == "tokenize") {
            fields(request, {"id", "op", "text", "parse_special", "add_special"});
            require(request.at("text").is_string(), "text must be a string");
            require(request.at("parse_special").is_boolean(), "parse_special must be boolean");
            require(request.at("add_special").is_boolean() && !request.at("add_special").get<bool>(), "add_special must be false");
            const auto text = request.at("text").get<std::string>();
            require(text.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()), "text too large");
            const bool parse_special = request.at("parse_special").get<bool>();
            int count = llama_tokenize(vocab_, text.data(), static_cast<int>(text.size()), nullptr, 0, false, parse_special);
            require(count != std::numeric_limits<int32_t>::min(), "tokenization overflow");
            if (count == 0) return {{"ids", json::array()}};
            require(count < 0, "unexpected tokenization sizing result");
            std::vector<llama_token> output(-count);
            count = llama_tokenize(vocab_, text.data(), static_cast<int>(text.size()), output.data(), static_cast<int>(output.size()), false, parse_special);
            require(count >= 0 && count <= static_cast<int>(output.size()), "tokenization failed");
            output.resize(count);
            return {{"ids", output}};
        }
        if (op == "token_pieces") {
            fields(request, {"id", "op", "ids"});
            auto ids = tokens(request.at("ids"), vocab_size_, true);
            std::vector<std::string> pieces;
            for (auto token : ids) {
                std::string bytes(128, '\0');
                int size = llama_token_to_piece(vocab_, token, bytes.data(), static_cast<int>(bytes.size()), 0, true);
                require(size != std::numeric_limits<int32_t>::min(), "token piece overflow");
                if (size < 0) {
                    bytes.resize(-size);
                    size = llama_token_to_piece(vocab_, token, bytes.data(), static_cast<int>(bytes.size()), 0, true);
                }
                require(size >= 0 && size <= static_cast<int>(bytes.size()), "token piece decoding failed");
                bytes.resize(size);
                pieces.push_back(hex(bytes));
            }
            return {{"pieces_hex", pieces}};
        }
        if (op == "score") {
            fields(request, {"id", "op", "input_ids", "score_positions"});
            const auto ids = tokens(request.at("input_ids"), vocab_size_);
            check_length(ids.size());
            const auto targets = positions(request.at("score_positions"), ids.size());
            std::set<int> sources;
            for (int target : targets) sources.insert(target - 1);
            std::vector<double> logprobs;
            double nll = 0;
            Batch batch;
            const int decode_count = targets.back();
            for (int offset = 0; offset < decode_count; offset += kBatch) {
                const int count = std::min(kBatch, decode_count - offset);
                decode(batch, ids, offset, count, sources);
                for (int local = 0; local < count; ++local) {
                    if (!sources.count(offset + local)) continue;
                    const auto logits = llama_get_logits_ith(context_.get(), local);
                    const double value = target_logprob(logits, vocab_size_, ids.at(offset + local + 1));
                    logprobs.push_back(value);
                    nll -= value;
                }
            }
            require(logprobs.size() == targets.size() && std::isfinite(nll), "invalid score result");
            return {{"target_logprobs", logprobs}, {"score_positions", targets}, {"nll_sum", nll}, {"count", targets.size()}};
        }
        if (op == "generate") {
            fields(request, {"id", "op", "input_ids", "max_new_tokens", "eos_ids"});
            auto ids = tokens(request.at("input_ids"), vocab_size_);
            const int maximum = static_cast<int>(integer(request.at("max_new_tokens"), 1, options_.context_length, "max_new_tokens"));
            check_length(ids.size() + static_cast<size_t>(maximum));
            const auto eos = tokens(request.at("eos_ids"), vocab_size_);
            const std::set<llama_token> eos_set(eos.begin(), eos.end());
            require(eos_set.size() == eos.size(), "eos_ids must be unique");
            std::vector<llama_token> generated;
            Batch batch;
            for (int step = 0; step < maximum; ++step) {
                clear();
                const int count_ids = static_cast<int>(ids.size());
                const std::set<int> outputs{count_ids - 1};
                for (int offset = 0; offset < count_ids; offset += kBatch) {
                    decode(batch, ids, offset, std::min(kBatch, count_ids - offset), outputs);
                }
                const auto token = argmax(llama_get_logits_ith(context_.get(), -1), vocab_size_);
                generated.push_back(token);
                if (eos_set.count(token)) return {{"generated_ids", generated}, {"stop_reason", "eos"}};
                ids.push_back(token);
            }
            return {{"generated_ids", generated}, {"stop_reason", "length"}};
        }
        throw std::runtime_error("unknown op: " + op);
    }
};

// Consume an overlong line fully, without retaining it, so the next request is independent.
bool read_line(std::istream & input, std::string & line, bool & overlong) {
    line.clear();
    overlong = false;
    char character;
    bool read_any = false;
    while (input.get(character)) {
        read_any = true;
        if (character == '\n') break;
        if (line.size() < kMaxLine) line.push_back(character);
        else overlong = true;
    }
    return read_any;
}

} // namespace prism_eval

#ifndef OPENBONSAI_WORKER_UNIT_TEST
int main(int argc, char ** argv) {
    using namespace prism_eval;
    try {
        const auto opts = options(argc, argv);
        llama_log_set([](ggml_log_level, const char * message, void *) { std::fputs(message, stderr); }, nullptr);
        llama_backend_init();
        {
            Worker worker(opts);
            std::string line;
            bool overlong = false;
            while (read_line(std::cin, line, overlong)) {
                json id = nullptr;
                json response;
                try {
                    require(!overlong, "request exceeds 16 MiB line limit");
                    const auto request = parse_request(line);
                    id = request.at("id");
                    response = worker.dispatch(request);
                    response["id"] = id;
                    response["ok"] = true;
                } catch (const std::exception & error) {
                    response = {{"id", id}, {"ok", false}, {"error", error.what()}};
                }
                std::cout << response.dump() << '\n' << std::flush;
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "prism-eval-worker: " << error.what() << '\n';
        return 1;
    }
}
#endif

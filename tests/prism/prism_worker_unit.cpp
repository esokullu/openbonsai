// Compiles the same parser/reduction/index validation code without loading weights.
#define OPENBONSAI_WORKER_UNIT_TEST
#include "../tools/prism_eval_worker.cpp"
#include <sstream>

using namespace prism_eval;
int checks = 0;
void check(bool condition) { require(condition, "unit assertion failed"); ++checks; }
template<typename Function> void rejects(Function action) {
    bool threw = false;
    try { action(); } catch (const std::exception &) { threw = true; }
    check(threw);
}

int main() {
    try {
        check(parse_request(R"({"id":1,"op":"metadata"})").at("id") == 1);
        check(parse_request(R"({"id":"a","op":"metadata"})").at("id") == "a");
        for (const std::string line : {R"({"id":1,"id":2,"op":"metadata"})", R"({"id":true,"op":"metadata"})",
                                      R"({"id":1.0,"op":"metadata"})", R"([])", R"({"id":1})", "null", "", "{} {}"}) {
            rejects([&] { parse_request(line); });
        }
        rejects([] { parse_request(R"({"id":1,"op":"score","x":{"a":1,"a":2}})"); });
        check(parse_request(R"({"id":1,"op":"score","x":[{"a":1},{"a":2}]})").is_object());
        const auto request = parse_request(R"({"id":1,"op":"metadata"})");
        fields(request, {"id", "op"}); ++checks;
        rejects([&] { fields(request, {"id", "op", "extra"}); });
        for (json bad : {json(true), json(1.0), json(-1), json(4), json("1"), json(nullptr), json(uint64_t(-1))}) {
            rejects([&] { integer(bad, 0, 3, "integer"); });
        }
        check(integer(0, 0, 3, "integer") == 0);
        check(integer(3, 0, 3, "integer") == 3);
        check(tokens(json::array({0, 3, 0}), 4) == std::vector<llama_token>({0, 3, 0}));
        check(tokens(json::array(), 4, true).empty());
        for (auto bad : {json::array(), json::array({4}), json::array({-1}), json::array({true}), json::array({1.0}), json(1)}) {
            rejects([&] { tokens(bad, 4); });
        }
        check(positions(json::array({1, 4, 65}), 66) == std::vector<int>({1, 4, 65}));
        for (auto bad : {json::array(), json::array({0}), json::array({2, 1}), json::array({1, 1}), json::array({4}), json::array({1.0})}) {
            rejects([&] { positions(bad, 4); });
        }
        const float uniform[] = {1000000000000.0f, 1000000000000.0f, 1000000000000.0f};
        check(std::abs(target_logprob(uniform, 3, 2) + std::log(3.0)) < 1e-14);
        const float row[] = {-1000, 1, 2, 2};
        const double reference = 1.0 - (2.0 + std::log(std::exp(-1.0) + 2.0));
        check(std::abs(target_logprob(row, 4, 1) - reference) < 1e-14);
        check(argmax(row, 4) == 2);
        check(argmax(uniform, 3) == 0);
        const float singleton[] = {std::numeric_limits<float>::max()};
        check(target_logprob(singleton, 1, 0) == 0);
        for (float invalid : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            const float values[] = {0, invalid};
            rejects([&] { target_logprob(values, 2, 0); });
            rejects([&] { argmax(values, 2); });
        }
        rejects([&] { target_logprob(row, 4, 4); });
        check(hex(std::string("\0\xff\xc3\xa9", 4)) == "00ffc3a9");
        std::istringstream stream("a\nb");
        std::string line; bool overlong;
        check(read_line(stream, line, overlong) && line == "a" && !overlong);
        check(read_line(stream, line, overlong) && line == "b" && !overlong);
        check(!read_line(stream, line, overlong));
        std::istringstream oversized(std::string(kMaxLine + 1, 'x') + "\nnext\n");
        check(read_line(oversized, line, overlong) && line.size() == kMaxLine && overlong);
        check(read_line(oversized, line, overlong) && line == "next" && !overlong);
        std::cout << json({{"ok", true}, {"checks", checks}, {"weights_loaded", false}}).dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

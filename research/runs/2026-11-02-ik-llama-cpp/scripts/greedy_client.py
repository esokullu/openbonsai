"""Send fixed raw prompts to a llama-server /completion endpoint with greedy decoding; save the replies.

Usage: python -I -X utf8 greedy_client.py URL OUT.json
"""
import json
import sys
import time
import urllib.request

CHAT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # Qwen3.5 template, thinking off

PROMPTS = [
    {"name": "arith", "n_predict": 16, "prompt": CHAT.format(q="What is 17 * 24? Reply with just the number.")},
    {"name": "word_problem", "n_predict": 128, "prompt": CHAT.format(
        q="A baker makes 48 cupcakes in the morning and half as many in the afternoon. "
          "How many cupcakes does the baker make in total? Answer briefly.")},
    {"name": "code", "n_predict": 128, "prompt": CHAT.format(
        q="Write a Python function fib(n) that returns the n-th Fibonacci number, with fib(0) = 0.")},
    {"name": "raw_text", "n_predict": 24, "prompt": "The Eiffel Tower is located in the city of"},
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    url, out = sys.argv[1].rstrip("/"), sys.argv[2]
    results = []
    for p in PROMPTS:
        body = {"prompt": p["prompt"], "n_predict": p["n_predict"], "temperature": 0.0, "top_k": 1,
                "seed": 0, "cache_prompt": False, "stream": False}
        t0 = time.time()
        r = post(url + "/completion", body)
        timings = r.get("timings", {})
        results.append({
            "name": p["name"], "content": r.get("content"), "tokens_predicted": r.get("tokens_predicted"),
            "stopped_eos": r.get("stopped_eos"), "stopped_limit": r.get("stopped_limit"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "prompt_per_second": timings.get("prompt_per_second"), "wall_s": round(time.time() - t0, 1),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

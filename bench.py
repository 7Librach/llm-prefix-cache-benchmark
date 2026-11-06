import argparse
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, asdict, field

SERVER_URL = "http://127.0.0.1:9931"

PREFIX_WORDS = (
    "system instruction context prefix token cache latency model inference "
    "processor memory device kernel attention layer embedding quantization "
    "benchmark measurement throughput sampling temperature sequence"
).split()

SUFFIX_WORDS = [
    "query", "answer", "result", "summary", "report", "detail",
    "aspect", "topic", "section", "paragraph", "list", "note",
]


def post_json(endpoint: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(endpoint: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(f"{SERVER_URL}{endpoint}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = get_json("/health", timeout=5)
            if h.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def tokenize(text: str) -> list[int]:
    res = post_json("/tokenize", {"content": text})
    return res["tokens"]


def make_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(PREFIX_WORDS) for _ in range(n_words))


def build_prefix(target_tokens: int, seed: int = 1234) -> str:
    lo, hi = 1, max(8, target_tokens * 3)
    best = make_text(hi, seed)
    for _ in range(40):
        mid = (lo + hi) // 2
        cand = make_text(mid, seed)
        n = len(tokenize(cand))
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
            best = cand
        if lo >= hi:
            break
    while len(tokenize(best)) < target_tokens:
        best += " " + PREFIX_WORDS[0]
    return best


def build_suffix(n_tokens: int, seed: int = 4321) -> str:
    rng = random.Random(seed)
    text = " ".join(rng.choice(SUFFIX_WORDS) for _ in range(max(4, n_tokens)))
    while len(tokenize(text)) < n_tokens:
        text += " " + rng.choice(SUFFIX_WORDS)
    return text


def lcp_length(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class Measurement:
    model: str
    prefix_tokens: int
    suffix_tokens: int
    cache_enabled: bool
    repeat: int
    prompt_ms: float
    predicted_ms: float
    total_ms: float
    prompt_n: int
    tokens_cached: int
    n_prompt_tokens: int
    prompt_per_second: float
    predicted_per_second: float
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_request(prompt: str, cache_prompt: bool, n_predict: int = 64,
                seed: int = 42, temperature: float = 0.0) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": cache_prompt,
        "seed": seed,
        "temperature": temperature,
        "top_k": 1,
        "stream": False,
        "timings_per_token": False,
    }
    return post_json("/completion", payload)


def measure_pair(prompt: str, cache_enabled: bool, model: str,
                 prefix_tokens: int, suffix_tokens: int, repeat: int,
                 n_predict: int, lcp_tokens: int = 0) -> Measurement:
    res = run_request(prompt, cache_enabled, n_predict=n_predict)
    t = res.get("timings", {})
    return Measurement(
        model=model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        cache_enabled=cache_enabled,
        repeat=repeat,
        prompt_ms=float(t.get("prompt_ms", 0.0)),
        predicted_ms=float(t.get("predicted_ms", 0.0)),
        total_ms=float(t.get("prompt_ms", 0.0)) + float(t.get("predicted_ms", 0.0)),
        prompt_n=int(t.get("prompt_n", 0)),
        tokens_cached=int(res.get("tokens_cached", 0)),
        n_prompt_tokens=int(res.get("tokens_evaluated", 0)),
        prompt_per_second=float(t.get("prompt_per_second", 0.0)),
        predicted_per_second=float(t.get("predicted_per_second", 0.0)),
        raw={
            "stop_type": res.get("stop_type"),
            "cache_n": int(t.get("cache_n", 0)),
            "predicted_n": int(t.get("predicted_n", 0)),
            "lcp_tokens": lcp_tokens,
            "content": res.get("content", ""),
        },
    )


def run_experiment(model_path: str, model_name: str,
                   prefix_lengths: list[int], suffix_tokens: int = 128,
                   repeats: int = 10, n_predict: int = 64,
                   out_path: str = "results/raw.json") -> None:
    if not wait_for_server():
        raise RuntimeError("Сервер не ответил на /health")

    results: list[Measurement] = []

    for p_len in prefix_lengths:
        prefix = build_prefix(p_len)
        real_prefix_tokens = len(tokenize(prefix))
        suffix_a = build_suffix(suffix_tokens, seed=4321)
        suffix_b = build_suffix(suffix_tokens, seed=8765)
        prompt_a = prefix + "\n\n" + suffix_a
        prompt_b = prefix + "\n\n" + suffix_b
        tok_a = tokenize(prompt_a)
        tok_b = tokenize(prompt_b)
        lcp_tok = lcp_length(tok_a, tok_b)
        print(f"p={real_prefix_tokens}, len(A)={len(tok_a)}, len(B)={len(tok_b)}, lcp={lcp_tok}", flush=True)

        # Прогрев перед замерами
        run_request(prompt_a, False, n_predict=8)
        run_request(prompt_b, True, n_predict=8)

        for rep in range(repeats):
            m_a = measure_pair(prompt_a, False, model_name, real_prefix_tokens,
                               suffix_tokens, rep, n_predict, lcp_tokens=lcp_tok)
            results.append(m_a)

            m_b = measure_pair(prompt_b, True, model_name, real_prefix_tokens,
                               suffix_tokens, rep, n_predict, lcp_tokens=lcp_tok)
            results.append(m_b)
            print(f"  [{rep}] no_cache={m_a.prompt_ms:.2f} ms, cache={m_b.prompt_ms:.2f} ms "
                  f"(cache_n={m_b.raw.get('cache_n')}, prompt_n={m_b.prompt_n})", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    print(f"Сохранено в {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--prefixes", default="128,256,512,1024,2048,4096")
    ap.add_argument("--suffix", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--out", default="results/raw.json")
    args = ap.parse_args()

    name = args.model_name or os.path.basename(args.model)
    lengths = [int(x) for x in args.prefixes.split(",") if x.strip()]

    run_experiment(args.model, name, lengths, args.suffix, args.repeats,
                   args.n_predict, args.out)


if __name__ == "__main__":
    main()

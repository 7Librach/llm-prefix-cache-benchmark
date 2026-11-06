import argparse
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, asdict

SERVER_URL = "http://127.0.0.1:9931"

WORDS = (
    "system instruction context prefix token cache latency model inference "
    "processor memory device kernel attention layer embedding quantization "
    "benchmark measurement throughput sampling temperature sequence"
).split()

SUFFIX_WORDS = [
    "query", "answer", "result", "summary", "report", "detail",
    "aspect", "topic", "section", "paragraph", "list", "note",
]


def post_json(ep: str, p: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        SERVER_URL + ep,
        data=json.dumps(p).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(ep: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(SERVER_URL + ep, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_server(timeout: int = 900) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if get_json("/health", timeout=5).get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def tokenize(text: str) -> list[int]:
    return post_json("/tokenize", {"content": text})["tokens"]


def make_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def build_prefix(target: int, seed: int = 1234) -> str:
    lo, hi = 1, max(8, target * 3)
    best = make_text(hi, seed)
    for _ in range(40):
        mid = (lo + hi) // 2
        cand = make_text(mid, seed)
        if len(tokenize(cand)) < target:
            lo = mid + 1
        else:
            hi = mid
            best = cand
        if lo >= hi:
            break
    while len(tokenize(best)) < target:
        best += " " + WORDS[0]
    return best


def build_suffix(target: int, seed: int = 4321) -> str:
    rng = random.Random(seed)
    t = " ".join(rng.choice(SUFFIX_WORDS) for _ in range(max(4, target)))
    while len(tokenize(t)) < target:
        t += " " + rng.choice(SUFFIX_WORDS)
    return t


@dataclass
class Row:
    model: str
    prefix_tokens: int
    suffix_tokens: int
    repeat: int
    prompt_ms_nocache: float
    prompt_ms_cache: float
    prompt_n_nocache: int
    prompt_n_cache: int
    saved_ms: float
    saved_tokens: int
    reduction_pct: float


def one(prompt: str, cache: bool, n_predict: int = 64) -> tuple[float, int]:
    r = post_json("/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": cache,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
    })
    t = r.get("timings", {})
    return float(t.get("prompt_ms", 0.0)), int(t.get("prompt_n", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--suffixes", default="32,64,128,256,512")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="results/raw_suffix.json")
    args = ap.parse_args()

    name = args.model_name or os.path.basename(args.model)
    suffixes = [int(x) for x in args.suffixes.split(",") if x.strip()]

    if not wait_for_server():
        raise RuntimeError("Сервер не готов")

    prefix = build_prefix(args.prefix)
    real_p = len(tokenize(prefix))

    rows: list[Row] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for s_len in suffixes:
        sfx_a = build_suffix(s_len, seed=4321)
        sfx_b = build_suffix(s_len, seed=8765)
        q1 = prefix + "\n\n" + sfx_a
        q2 = prefix + "\n\n" + sfx_b

        one(q1, False, 8)
        one(q2, True, 8)

        for rep in range(args.repeats):
            ms_a, n_a = one(q1, False)
            ms_b, n_b = one(q2, True)
            rows.append(Row(
                model=name,
                prefix_tokens=real_p,
                suffix_tokens=s_len,
                repeat=rep,
                prompt_ms_nocache=ms_a,
                prompt_ms_cache=ms_b,
                prompt_n_nocache=n_a,
                prompt_n_cache=n_b,
                saved_ms=ms_a - ms_b,
                saved_tokens=n_a - n_b,
                reduction_pct=100.0 * (ms_a - ms_b) / ms_a if ms_a else 0.0,
            ))
        print(f"suffix={s_len}: reduction={rows[-1].reduction_pct:.1f}%, saved={rows[-1].saved_ms:.1f} ms", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

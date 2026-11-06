import json
import os
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/figures"

MODELS = [
    ("1.5B", "Qwen2.5-1.5B-Instruct-Q4_K_M", "results/raw_1_5b_v2.json", "#1f77b4"),
    ("3B",   "Qwen2.5-3B-Instruct-Q4_K_M",   "results/raw_3b_v2.json",   "#d62728"),
    ("7B",   "Qwen2.5-7B-Instruct-Q4_K_M",   "results/raw_7b_v2.json",   "#2ca02c"),
]


def load_series() -> pd.DataFrame:
    frames = []
    for short, label, path, _ in MODELS:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["model_short"] = short
        df["model_label"] = label
        df["cache_n"] = df["raw"].apply(lambda r: r.get("cache_n", 0) if isinstance(r, dict) else 0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    no_cache = df[~df["cache_enabled"]].copy()
    cache = df[df["cache_enabled"]].copy()

    merged = pd.merge(
        no_cache,
        cache,
        on=["model_short", "model_label", "prefix_tokens", "suffix_tokens", "repeat"],
        suffixes=("_nocache", "_cache"),
    )
    merged["saved_ms"] = merged["prompt_ms_nocache"] - merged["prompt_ms_cache"]
    merged["reduction_pct"] = 100.0 * merged["saved_ms"] / merged["prompt_ms_nocache"]

    rows = []
    stats_out = {}

    for short, label, _, _ in MODELS:
        sub = merged[merged["model_short"] == short]
        w_stat, w_p = stats.wilcoxon(sub["prompt_ms_nocache"], sub["prompt_ms_cache"], alternative="greater")
        lr = stats.linregress(sub["prefix_tokens"], sub["prompt_ms_nocache"])
        lr_cache = stats.linregress(sub["prefix_tokens"], sub["prompt_ms_cache"])

        stats_out[short] = {
            "wilcoxon_W": float(w_stat),
            "wilcoxon_p": float(w_p),
            "ols_slope_ms_per_tok": float(lr.slope),
            "ols_slope_stderr": float(lr.stderr),
            "ols_intercept_ms": float(lr.intercept),
            "ols_r2": float(lr.rvalue ** 2),
            "cache_slope_ms_per_tok": float(lr_cache.slope),
        }

        for p_len, grp in sub.groupby("prefix_tokens"):
            rows.append({
                "model": short,
                "prefix_tokens": int(p_len),
                "cache_n": int(grp["cache_n_cache"].iloc[0]),
                "prompt_n_cache": int(grp["prompt_n_cache"].iloc[0]),
                "t_nocache_mean": float(grp["prompt_ms_nocache"].mean()),
                "t_nocache_std": float(grp["prompt_ms_nocache"].std(ddof=1)),
                "t_cache_mean": float(grp["prompt_ms_cache"].mean()),
                "t_cache_std": float(grp["prompt_ms_cache"].std(ddof=1)),
                "saved_ms_mean": float(grp["saved_ms"].mean()),
                "reduction_pct_mean": float(grp["reduction_pct"].mean()),
                "reduction_pct_std": float(grp["reduction_pct"].std(ddof=1)),
            })

    return pd.DataFrame(rows), stats_out


def plot_figures(summary: pd.DataFrame) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for short, label, _, color in MODELS:
        s = summary[summary["model"] == short].sort_values("prefix_tokens")
        ax.errorbar(
            s["prefix_tokens"], s["t_nocache_mean"], yerr=s["t_nocache_std"],
            marker="o", markersize=5.5, capsize=3, linewidth=1.8, color=color,
            label=f"{label} (без кэша)",
        )
        ax.errorbar(
            s["prefix_tokens"], s["t_cache_mean"], yerr=s["t_cache_std"],
            marker="s", markersize=5.5, capsize=3, linewidth=1.8, linestyle="--", color=color,
            label=f"{label} (с кэшем)",
        )

    ax.set_xlabel("Длина общего префикса, токенов", fontsize=11)
    ax.set_ylabel("Время обработки запроса, мс", fontsize=11)
    ax.set_title("Время обработки входного запроса без кэширования и с кэшированием", fontsize=11.5)
    ax.set_xlim(0, 4300)
    ax.set_ylim(-40, 1950)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_prompt_ms_ru.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for short, label, _, color in MODELS:
        s = summary[summary["model"] == short].sort_values("prefix_tokens")
        ax.errorbar(
            s["prefix_tokens"], s["reduction_pct_mean"], yerr=s["reduction_pct_std"],
            marker="o", markersize=5.5, capsize=3, linewidth=1.8, color=color,
            label=label,
        )

    ax.set_xlabel("Длина общего префикса, токенов", fontsize=11)
    ax.set_ylabel("Относительное сокращение времени, %", fontsize=11)
    ax.set_title("Относительное сокращение времени обработки запроса при кэшировании", fontsize=11.5)
    ax.set_xlim(0, 4300)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_reduction_pct_ru.png"), dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_series()
    summary, stats_out = summarize(df)
    summary.to_csv("results/summary_v2.csv", index=False)
    with open("results/stats_v2.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)
    plot_figures(summary)
    print("Готово: results/summary_v2.csv, results/stats_v2.json, графики в results/figures/")


if __name__ == "__main__":
    main()

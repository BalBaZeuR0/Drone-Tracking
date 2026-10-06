"""Deney sonuç defteri: her koşunun ``results.json`` dosyasından bölge başına tek satır.

Kullanım:
    python -m dronetrackingrl.real_data.ledger add <koşu_klasörü> --name <ad>
    python -m dronetrackingrl.real_data.ledger table [--names a b c]
    python -m dronetrackingrl.real_data.ledger plot --names a b c --out figures/x.png

``add`` koşunun ``results.json`` ve ``summary.txt`` dosyasını ``results/<ad>/`` altına
kopyalar ve ``results/ledger.csv``'ye satır ekler (aynı ad+bölge varsa değiştirir).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

COLUMNS = [
    "name", "split", "ref_start", "ref_end", "steps", "obs_mode", "switch_penalty", "include_own_history",
    "train_cache", "eval_cache", "oracle", "random_mean", "random_std", "best_fixed_cam", "best_fixed",
    "ppo_mean", "ppo_std", "ppo_seed_rates", "switches_mean", "seeds",
]
DEFAULT_RESULTS = Path(__file__).resolve().parents[3] / "results"


def rows_from_results(results: dict, name: str) -> List[dict]:
    """Bir ``results.json`` sözlüğünden, bölge başına bir satır üretir."""
    config = results["config"]
    rows = []
    for split, info in results["splits"].items():
        base = results["baselines"][split]
        runs = results["ppo"].values()
        rates = [run["splits"][split]["visible_rate"] for run in runs]
        switches = [run["splits"][split]["switches"] for run in runs]
        rows.append({
            "name": name,
            "split": split,
            "ref_start": info["ref_start"],
            "ref_end": info["ref_end"],
            "steps": info["steps"],
            "obs_mode": config.get("obs_mode", "latent"),
            "switch_penalty": config.get("switch_penalty", 0.0),
            "include_own_history": bool(config.get("include_own_history", False)),
            "train_cache": Path(config["train_cache"]).name,
            "eval_cache": Path(config.get("eval_cache") or config["train_cache"]).name,
            "oracle": round(base["oracle"]["visible_rate"], 4),
            "random_mean": round(base["random"]["visible_rate_mean"], 4),
            "random_std": round(base["random"]["visible_rate_std"], 4),
            "best_fixed_cam": base["best_fixed"]["camera"],
            "best_fixed": round(base["best_fixed"]["visible_rate"], 4),
            "ppo_mean": round(float(np.mean(rates)), 4),
            "ppo_std": round(float(np.std(rates)), 4),
            "ppo_seed_rates": ";".join(f"{r:.4f}" for r in rates),
            "switches_mean": round(float(np.mean(switches)), 1),
            "seeds": len(rates),
        })
    return rows


def read_ledger(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_ledger(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def add_run(run_dir: Path, name: str, results_dir: Path = DEFAULT_RESULTS) -> int:
    """Koşuyu defterin klasörüne kopyalar ve satırlarını ekler. Eklenen satır sayısını döndürür."""
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    target = results_dir / name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "results.json", target / "results.json")
    if (run_dir / "summary.txt").exists():
        shutil.copy(run_dir / "summary.txt", target / "summary.txt")
    new_rows = rows_from_results(results, name)
    ledger_path = results_dir / "ledger.csv"
    kept = [row for row in read_ledger(ledger_path) if row["name"] != name]
    write_ledger(kept + new_rows, ledger_path)
    return len(new_rows)


def table_markdown(rows: List[dict], names: Optional[List[str]] = None) -> str:
    selected = [r for r in rows if names is None or r["name"] in names]
    header = "| koşu | bölge | oracle | sabit (cam) | PPO | geçiş | tohum |\n|---|---|---|---|---|---|---|"
    lines = [header]
    for r in selected:
        lines.append(
            f"| {r['name']} | {r['split']} ({r['ref_start']}–{r['ref_end']}) | "
            f"{float(r['oracle']) * 100:.1f} | {float(r['best_fixed']) * 100:.1f} (cam{r['best_fixed_cam']}) | "
            f"{float(r['ppo_mean']) * 100:.1f} ± {float(r['ppo_std']) * 100:.1f} | "
            f"{float(r['switches_mean']):.0f} | {r['seeds']} |"
        )
    return "\n".join(lines)


def plot_comparison(rows: List[dict], names: List[str], out: Path) -> Path:
    """Seçilen koşuların bölge bazında PPO ortalamasını (hata çubuğu ile) sabit kamera ve oracle ile birlikte çizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in rows if r["name"] in names]
    splits = list(dict.fromkeys(r["split"] for r in selected))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(splits)), 5))
    width = 0.8 / max(len(names), 1)
    xs = np.arange(len(splits))
    for k, name in enumerate(names):
        ys, errs = [], []
        for split in splits:
            match = [r for r in selected if r["name"] == name and r["split"] == split]
            ys.append(float(match[0]["ppo_mean"]) * 100 if match else np.nan)
            errs.append(float(match[0]["ppo_std"]) * 100 if match else 0)
        ax.bar(xs + (k - (len(names) - 1) / 2) * width, ys, width, yerr=errs, capsize=3, label=f"PPO: {name}")
    fixed = [float(next(r["best_fixed"] for r in selected if r["split"] == s)) * 100 for s in splits]
    oracle = [float(next(r["oracle"] for r in selected if r["split"] == s)) * 100 for s in splits]
    ax.scatter(xs, fixed, marker="_", s=600, color="black", zorder=3, label="en iyi sabit kamera")
    ax.scatter(xs, oracle, marker="^", color="#14947f", zorder=3, label="oracle")
    ax.set_xticks(xs)
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel("görünür oranı (%)")
    ax.set_ylim(40, 102)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dronetrackingrl.real_data.ledger")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="bir koşuyu deftere ekle")
    add.add_argument("run_dir")
    add.add_argument("--name", required=True)
    table = sub.add_parser("table", help="markdown tablo yazdır")
    table.add_argument("--names", nargs="*")
    plot = sub.add_parser("plot", help="karşılaştırma grafiği")
    plot.add_argument("--names", nargs="+", required=True)
    plot.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    if args.command == "add":
        count = add_run(Path(args.run_dir), args.name, results_dir)
        print(f"{count} satır eklendi: {args.name}")
    elif args.command == "table":
        print(table_markdown(read_ledger(results_dir / "ledger.csv"), args.names))
    elif args.command == "plot":
        print("kaydedildi:", plot_comparison(read_ledger(results_dir / "ledger.csv"), args.names, Path(args.out)))


if __name__ == "__main__":
    main()

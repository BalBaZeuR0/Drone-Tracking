"""README result figures, drawn from the saved results/*/results.json files (no retraining).

    python scripts/plot_readme.py            # -> plots/same_scene_heldout.png, plots/new_scene.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIXED, ORACLE, INK, MUTED, GRID = "#9a9a94", "#2b2b28", "#2b2b28", "#6b6b66", "#e6e5e0"
SERIES = {"PPO (classical detector)": "#2a78d6", "PPO (RELO detector)": "#eb6834"}


def load(run: str) -> dict:
    return json.loads((ROOT / "results" / run / "results.json").read_text(encoding="utf-8"))


def split_row(d: dict, split: str) -> dict:
    sp, base = d["splits"][split], d["baselines"][split]
    seeds = [d["ppo"][s]["splits"][split]["visible_rate"] * 100 for s in sorted(d["ppo"])]
    return {"region": f"ds{'3' if len(sp['cameras']) == 6 else '4'} {sp['ref_start']}–{sp['ref_end']}",
            "fixed": base["best_fixed"]["visible_rate"] * 100, "oracle": base["oracle"]["visible_rate"] * 100,
            "seeds": seeds, "mean": sum(seeds) / len(seeds)}


def dot_plot(rows: list[dict], series: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 0.95 * len(rows) + 1.5), dpi=150)
    n = len(series)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["fixed"]], [y], "o", ms=9, color=FIXED, mec="white", mew=1.5, zorder=3)
        ax.plot([r["oracle"]], [y], "|", ms=18, mew=2.5, color=ORACLE, zorder=3)
        for j, name in enumerate(series):
            s = r[name]
            yy = y + (j - (n - 1) / 2) * 0.22
            ax.plot([min(s["seeds"]), max(s["seeds"])], [yy, yy], color=SERIES[name], lw=2, alpha=0.45, zorder=2)
            ax.plot([s["mean"]], [yy], "o", ms=9, color=SERIES[name], mec="white", mew=1.5, zorder=4)
            d = s["mean"] - r["fixed"]
            ax.text(max(s["seeds"] + [r["oracle"]]) + 0.8, yy, f"{s['mean']:.1f}  ({d:+.1f})", va="center",
                    fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r["region"] for r in reversed(rows)], fontsize=9.5, color=INK)
    ax.set_xlabel("Visible rate (%): is the drone visible in the camera the agent picked?", fontsize=9, color=MUTED)
    lo = min(min(r["fixed"], *(min(r[s]["seeds"]) for s in series)) for r in rows)
    ax.set_xlim(max(0, lo - 4), 110)
    ax.set_xticks([t for t in ax.get_xticks() if max(0, lo - 4) <= t <= 100])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=FIXED, label="Best fixed camera"),
               *[plt.Line2D([], [], ls="", marker="o", ms=8, color=SERIES[s], label=f"{s}, 3 seeds") for s in series],
               plt.Line2D([], [], ls="", marker="|", ms=12, mew=2.5, color=ORACLE, label="Oracle (upper bound)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles) if len(handles) <= 3 else 2, frameon=False,
              fontsize=8.5, handletextpad=0.3, columnspacing=1.2)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=28 if len(handles) <= 3 else 44)
    fig.text(0.01, 0.005, "Dot: seed mean · line: seed range · in brackets: difference from the fixed camera (points)",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(out)


def main() -> None:
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    c = load("C_hist")
    rows = []
    for split in ("eval", "eval_extra1", "eval_extra2", "eval_extra3"):
        r = split_row(c, split)
        r["PPO (classical detector)"] = {"mean": r["mean"], "seeds": r["seeds"]}
        rows.append(r)
    dot_plot(rows, ["PPO (classical detector)"],
             "Held-out segments of the training scenes (C_hist: mixed training + own history)",
             out / "same_scene_heldout.png")

    rows = []
    for direction, label in (("ds3_to_ds4", "ds3 → ds4"), ("ds4_to_ds3", "ds4 → ds3")):
        k, rl = split_row(load(f"X_{direction}_fixed"), "eval"), split_row(load(f"X_{direction}_relo_ozelliksiz"), "eval")
        rows.append({"region": f"{label}\n(test: {k['region']})", "fixed": k["fixed"], "oracle": k["oracle"],
                     "PPO (classical detector)": {"mean": k["mean"], "seeds": k["seeds"]},
                     "PPO (RELO detector)": {"mean": rl["mean"], "seeds": rl["seeds"]}})
    dot_plot(rows, list(SERIES), "Transfer to an unseen scene (no target-scene frames in training)",
             out / "new_scene.png")


if __name__ == "__main__":
    main()

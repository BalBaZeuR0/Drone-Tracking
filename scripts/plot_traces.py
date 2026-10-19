"""README trace figures, drawn from a run's per-step trace files (no retraining).

    python scripts/plot_traces.py --trace-dir outputs/C_hist/seed_0 --data-dir DATA
    # -> assets/camera_selection.png, plots/visible_rate_over_time.png

--data-dir is the drone-tracking-datasets checkout (needs datasetN/cam0.jpg and datasetN/detections/cam0.txt).
cam0 is the reference camera, so its frame number is the ref_frame used in the traces.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dronetrackingrl.real_data.detections import parse_detections_file  # noqa: E402

CHOSEN, SEEN, FIXED, ORACLE = "#d9560f", "#cfd4dc", "#3b5a80", "#5fb3a1"
# trace file -> (dataset, held-out?) for the C_hist split layout (see README "Train the C_hist agent")
TRACES = {
    "trace_train.csv": (3, False), "trace_train_extra1.csv": (3, False), "trace_train_extra2.csv": (3, False),
    "trace_train_extra3.csv": (4, False), "trace_train_extra4.csv": (4, False), "trace_train_extra5.csv": (4, False),
    "trace_eval.csv": (3, True), "trace_eval_extra1.csv": (3, True),
    "trace_eval_extra2.csv": (4, True), "trace_eval_extra3.csv": (4, True),
}
N_CAMS = {3: 6, 4: 7}


def load_traces(trace_dir: Path) -> tuple[dict, dict]:
    rows = {3: [], 4: []}
    held_out = {3: [], 4: []}
    for name, (ds, heldout) in TRACES.items():
        with open(trace_dir / name, newline="") as f:
            part = [r for r in csv.DictReader(f)]
        rows[ds] += part
        if heldout:
            held_out[ds].append((int(part[0]["ref_frame"]), int(part[-1]["ref_frame"])))
    for ds in rows:
        rows[ds].sort(key=lambda r: int(r["ref_frame"]))
    return rows, held_out


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] == "True" for r in rows])


def camera_selection(rows: dict, data_dir: Path, out: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.15]})
    for row, ds in enumerate((3, 4)):
        ax = axs[row, 0]
        ax.imshow(plt.imread(data_dir / f"dataset{ds}" / "cam0.jpg").mean(axis=2), cmap="gray")
        det = parse_detections_file(str(data_dir / f"dataset{ds}" / "detections" / "cam0.txt"))
        pts = np.array([(k, *v) for k, v in sorted(det.items()) if v])
        sc = ax.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap="plasma", s=2, lw=0)
        ax.set_title(f"dataset{ds} / cam0: ground-truth 2D pixel trajectory ({len(pts)} frames)", fontsize=10)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("time (ref_frame)")

        ax = axs[row, 1]
        t = np.array([int(r["ref_frame"]) for r in rows[ds]])
        cams = np.array([int(r["camera"]) for r in rows[ds]])
        for c in range(N_CAMS[ds]):
            seen = column(rows[ds], f"visible_cam{c}")
            ax.scatter(t[seen], np.full(seen.sum(), c), s=4, marker="|", color=SEEN, lw=1)
            pick = cams == c
            ax.scatter(t[pick], np.full(pick.sum(), c), s=4, marker="|", color=CHOSEN, lw=1,
                       label="agent's choice" if c == 0 else None)
        ax.set_yticks(range(N_CAMS[ds]), [f"cam{c}" for c in range(N_CAMS[ds])])
        ax.set_title(f"dataset{ds}: cameras that saw the drone (grey) vs the agent's choice (orange)", fontsize=10)
        ax.set_xlabel("ref_frame (time)")
        ax.set_ylabel("camera")
        ax.legend(loc="upper right", markerscale=3, fontsize=8.5)
    fig.suptitle("DroneTrackingRL: dataset3 & dataset4 (C_hist agent, seed 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(out)


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w)
    return np.convolve(x.astype(float), k, mode="same") / np.convolve(np.ones(len(x)), k, mode="same") * 100


def visible_over_time(rows: dict, held_out: dict, out: Path, window: int = 500) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.6), dpi=150)
    for ax, ds in zip(axs, (3, 4)):
        t = np.array([int(r["ref_frame"]) for r in rows[ds]])
        for a, b in held_out[ds]:
            ax.axvspan(a, b, color="#d0d0d0", alpha=0.5, lw=0)
        ax.fill_between(t, rolling(column(rows[ds], "any_visible"), window), color=ORACLE, alpha=0.2,
                        label="oracle (upper bound)")
        ax.plot(t, rolling(column(rows[ds], "visible_cam0"), window), color=FIXED, lw=1.2, label="fixed cam0")
        ax.plot(t, rolling(column(rows[ds], "chosen_visible"), window), color=CHOSEN, lw=1.6, label="PPO (C_hist)")
        ax.set_ylim(0, 105)
        ax.set_title(f"dataset{ds}", fontsize=11)
        ax.set_xlabel("ref_frame (time)")
        ax.set_ylabel("visible rate (%)")
        ax.legend(loc="lower center", ncol=3, fontsize=8.5)
        ax.text(0.99, 0.06, "grey band = segment never seen in training (held-out)", transform=ax.transAxes,
                ha="right", fontsize=7.5, color="#666666")
    fig.suptitle(f"Accuracy against ground truth: C_hist agent vs oracle vs fixed camera "
                 f"(seed 0, rolling window = {window} frames)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, default=ROOT / "outputs" / "C_hist" / "seed_0")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "DATA")
    args = ap.parse_args()
    rows, held_out = load_traces(args.trace_dir)
    camera_selection(rows, args.data_dir, ROOT / "assets" / "camera_selection.png")
    visible_over_time(rows, held_out, ROOT / "plots" / "visible_rate_over_time.png")


if __name__ == "__main__":
    main()

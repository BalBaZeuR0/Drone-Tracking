"""(a) Hayalet kamera seçimleri: kaç ayrı bölüm, en uzunu, ilk seçim karesi.
(b) X_ds3_to_ds4_relo: kaçırılan karelerde seçilen kameranın RELO/klasik durumu
(drone görünmezken RELO takipte mi, güveni ne?) — sağlıklı tohumla karşılaştırma."""
import csv
import sys

import numpy as np

from dronetrackingrl.real_data.experiment import SignalCache


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [r[k] for r in rows] for k in rows[0]}


out = sys.argv[1]
print("(a) hayalet kamera bölümleri, foldB_relo_s345 tohum 5")
for region in ("trace_train", "trace_eval", "trace_eval_extra2"):
    t = load(f"{out}/foldB_relo_s345/seed_5/{region}.csv")
    ghost = np.array([int(c) < 0 for c in t["camera"]])
    anyv = np.array([v == "True" for v in t["any_visible"]])
    starts = np.flatnonzero(ghost & ~np.r_[False, ghost[:-1]])
    lengths = []
    for s in starts:
        e = s
        while e + 1 < len(ghost) and ghost[e + 1]:
            e += 1
        lengths.append(e - s + 1)
    print(f"  {region}: {len(starts)} bölüm, uzunluklar {sorted(lengths, reverse=True)[:6]}, ilk {starts[:4].tolist()}, "
          f"hayaletteyken başka kamera görünür %{anyv[ghost].mean() * 100:.0f}")

print("\n(b) X_ds3_to_ds4_relo, ds4 1-26000: kaçırılan karelerde seçilen kamera")
cache = SignalCache.load(sys.argv[2]).slice(1, 26000)
for seed in (0, 1, 2):
    t = load(f"{out}/X_ds3_to_ds4_relo/seed_{seed}/trace_eval.csv")
    ref = np.array([int(r) for r in t["ref_frame"]]) - cache.ref_start
    cam = np.array([int(c) for c in t["camera"]])
    vis = np.array([v == "True" for v in t["chosen_visible"]])
    anyv = np.array([v == "True" for v in t["any_visible"]])
    j = np.array([cache.cameras.index(c) for c in cam])
    tracking = cache.relo_tracking[ref, j]
    score = np.nan_to_num(cache.relo_score[ref, j], nan=0.0)
    fired = ~np.isnan(cache.mask_pixel[ref, j, 0])
    miss = ~vis & anyv
    hit = vis
    print(f"  tohum {seed}: kaçan {miss.sum():5d} kare | kaçanda seçilen kamera: RELO takipte %{tracking[miss].mean() * 100:.0f}, "
          f"güven medyan {np.median(score[miss]):.2f}, klasik ateşledi %{fired[miss].mean() * 100:.0f} "
          f"|| isabette: takipte %{tracking[hit].mean() * 100:.0f}, güven {np.median(score[hit]):.2f}, "
          f"ateşledi %{fired[hit].mean() * 100:.0f}")

# Kamera başına RELO yanlış pozitif oranı (drone görünmezken takipte)
print("\n  ds4 1-26000, kamera başına: drone görünmezken RELO takipte % / klasik ateşledi %")
for j, c in enumerate(cache.cameras):
    absent = cache.recording[:, j] & ~cache.visible[:, j]
    print(f"    cam{c}: görünmez kare {absent.sum():5d}, RELO takipte %{cache.relo_tracking[absent, j].mean() * 100:.0f}, "
          f"klasik ateşledi %{(~np.isnan(cache.mask_pixel[absent, j, 0])).mean() * 100:.0f}")

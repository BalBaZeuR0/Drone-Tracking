"""Dedektör düzeyi kapı: klasik önbellek dilimi vs RELO hibrit önbelleği, aynı aralık.
Kullanım: python relo_gate.py KLASIK.npz RELO.npz [RELO2.npz ...]"""
import sys

import numpy as np

from dronetrackingrl.real_data.experiment import SignalCache, observation_diagnostics

HIT_PX = 10.0  # tespit, etiketin bu kadar px yakınındaysa "doğru yerde"


def summarize(cache):
    diag = observation_diagnostics(cache)
    rows = {}
    for j, cam in enumerate(cache.cameras):
        d = diag[str(cam)]
        gt = cache.visible[:, j]
        det = ~np.isnan(cache.mask_pixel[:, j, 0])
        both = gt & det
        err = np.linalg.norm(cache.mask_pixel[both, j] - cache.gt_pixel[both, j], axis=1)
        right_place = int((err <= HIT_PX).sum())
        right_20 = int((err <= 20.0).sum())
        rows[cam] = dict(
            prec=d["precision"], rec=d["recall"], med=d["centroid_error_px_median"],
            hit=right_place / max(int(gt.sum()), 1),  # etiketli karelerin doğru yerde bulunan oranı
            hit20=right_20 / max(int(gt.sum()), 1),
            trk=float(cache.relo_tracking[:, j].mean()),
        )
    return rows


def fmt(v):
    return "  -  " if v is None else f"{v:5.3f}" if v < 10 else f"{v:5.1f}"


classic = SignalCache.load(sys.argv[1])
relos = [SignalCache.load(p) for p in sys.argv[2:]]
start, end = relos[0].ref_start, relos[0].ref_end
classic = classic.slice(start, end)
results = [("klasik", summarize(classic))] + [(p.split("/")[-1], summarize(c)) for p, c in zip(sys.argv[2:], relos)]
print(f"aralık {start}-{end}; hit = etiketli karelerde <= {HIT_PX:g} px tespit oranı; trk = RELO takipte oranı")
for cam in classic.cameras:
    for name, rows in results:
        r = rows[cam]
        print(f"cam{cam} {name:>40s}  prec {fmt(r['prec'])}  rec {fmt(r['rec'])}  hit {fmt(r['hit'])}"
              f"  hit20 {fmt(r['hit20'])}  medpx {fmt(r['med'])}  trk {r['trk']:.2f}")
    print()
for name, rows in results:
    hits = [r["hit"] for r in rows.values()]
    precs = [r["prec"] for r in rows.values() if r["prec"] is not None]
    hits20 = [r["hit20"] for r in rows.values()]
    print(f"ORTALAMA {name:>40s}  hit {np.mean(hits):.3f}  hit20 {np.mean(hits20):.3f}  prec {np.mean(precs):.3f}")

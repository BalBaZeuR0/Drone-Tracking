"""RELO pilotunun kare-kare teşhisi: başlatma anı, kayma, güvenin ayırt ediciliği."""
import sys

import numpy as np

from dronetrackingrl.real_data.experiment import SignalCache

classic = SignalCache.load(sys.argv[1])
relo = SignalCache.load(sys.argv[2])
classic = classic.slice(relo.ref_start, relo.ref_end)

for j, cam in enumerate(relo.cameras):
    trk = relo.relo_tracking[:, j]
    gt = relo.visible[:, j]
    err = np.linalg.norm(relo.mask_pixel[:, j] - relo.gt_pixel[:, j], axis=1)  # NaN: etiket/tespit yok
    cerr = np.linalg.norm(classic.mask_pixel[:, j] - classic.gt_pixel[:, j], axis=1)
    starts = np.flatnonzero(trk & ~np.r_[False, trk[:-1]])
    print(f"--- cam{cam}: {len(starts)} başlatma, ilk başlatma karesi {starts[:5].tolist()}")
    for s in starts[:3]:
        window = slice(s, min(s + 400, len(trk)))
        e = err[window]
        print(f"  başlatma @{s}: başlatma anı hata {err[s]:.1f}px (klasik {cerr[s]:.1f}),"
              f" +10 {np.nanmedian(e[1:11]):.1f}, +50 {np.nanmedian(e[40:60]):.1f}, +200 {np.nanmedian(e[190:210]):.1f}")
    good = trk & gt & (err <= 10)
    bad = trk & gt & (err > 50)
    off = trk & ~gt
    score = relo.relo_score[:, j]
    for name, m in (("doğru(<=10px)", good), ("yanlış(>50px)", bad), ("etiketsiz", off)):
        if m.any():
            q = np.nanpercentile(score[m], [10, 50, 90])
            wh = np.nanmedian(relo.relo_box_wh[m, j], axis=0)
            print(f"  güven {name:14s} n={int(m.sum()):5d}  p10/50/90 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  kutu medyan {wh[0]:.0f}x{wh[1]:.0f}")
    # klasik dedektör RELO'dan uzakta (>50px) ve etikete yakın (<=10px) mı tespit ediyor?
    dist = np.linalg.norm(relo.mask_pixel[:, j] - classic.mask_pixel[:, j], axis=1)
    disagree = trk & (dist > 50)
    classic_right = disagree & (cerr <= 10)
    print(f"  RELO takipteyken klasik >50px uzakta: {int(disagree.sum())} kare; bunların {int(classic_right.sum())}'inde klasik doğru")

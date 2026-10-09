"""RELO sabit noktaya nasıl yapışıyor? Her 'sabit noktaya giriş' anını sınıflandırır:
BAŞLATMA (o kare bir init/yeniden çapa: takipte ama skor yok) ya da KAYMA (takip
sürerken RELO oraya gitti). Kayma ise önceki karelerde drone nerede/görünür müydü."""
import sys

import numpy as np

from dronetrackingrl.real_data.experiment import SignalCache

relo = SignalCache.load(sys.argv[1])
classic = SignalCache.load(sys.argv[2]).slice(relo.ref_start, relo.ref_end)
j = relo.cameras.index(int(sys.argv[3]))
point = np.array([float(sys.argv[4]), float(sys.argv[5])])

pos = relo.mask_pixel[:, j]
at_point = relo.relo_tracking[:, j] & (np.linalg.norm(pos - point, axis=1) <= 10)
init_frame = relo.relo_tracking[:, j] & np.isnan(relo.relo_score[:, j])
entries = np.flatnonzero(at_point & ~np.r_[False, at_point[:-1]])
print(f"cam{relo.cameras[j]} {relo.ref_start}-{relo.ref_end}: sabit nokta {point.tolist()}, "
      f"orada geçen kare {int(at_point.sum())}, giriş sayısı {len(entries)}")
counts = {"BAŞLATMA": 0, "KAYMA": 0}
for e in entries:
    kind = "BAŞLATMA" if init_frame[e] else "KAYMA"
    counts[kind] += 1
    end = e
    while end + 1 < len(at_point) and at_point[end + 1]:
        end += 1
    line = f"  @{e:5d} {kind:8s} süre {end - e + 1:5d} kare"
    if kind == "KAYMA":
        back = slice(max(e - 10, 0), e)
        gt_vis = relo.visible[back, j]
        relo_to_gt = np.linalg.norm(pos[back] - relo.gt_pixel[back, j], axis=1)
        gt_to_point = np.linalg.norm(relo.gt_pixel[back, j] - point, axis=1)
        line += (f" | önceki 10 karede drone görünür {int(gt_vis.sum())}/10, RELO-drone mesafesi medyan "
                 f"{np.nanmedian(relo_to_gt) if gt_vis.any() else float('nan'):.0f}px, drone-anten mesafesi "
                 f"{np.nanmin(gt_to_point) if gt_vis.any() else float('nan'):.0f}px, kutu "
                 f"{np.round(relo.relo_box_wh[e - 1, j]).tolist()}")
    else:
        line += f" | başlatan klasik tespit {np.round(classic.mask_pixel[e, j]).tolist()}"
    print(line)
print("  özet:", counts)

# DroneTrackingRL

Sabit, farklı açılardan bakan birden çok kamera arasında, o an drone'u gösteren
kamerayı seçmeyi öğrenen bir RL (PPO / Gymnasium) ajanı. Veri:
[CenekAlbl/drone-tracking-datasets](https://github.com/CenekAlbl/drone-tracking-datasets).

- **Kurulum ve komutlar:** [RUN_ON_OTHER_PC.md](RUN_ON_OTHER_PC.md)
- **Deney sonuçları (başarılı ve başarısız denemeler dahil):** [EXPERIMENTS.md](EXPERIMENTS.md)
- **Grafikli durum raporu (2026-09-21):** https://claude.ai/artifact/3m6Zoe1PMFg2cngrH6ky9W

## Kısaca

```
python -m dronetrackingrl.real_data.prepare extract --dataset-dir ... --frames-dir ...   # 1) kareleri çıkar
python -m dronetrackingrl.real_data.experiment build-cache --sync dataset3 ...           # 2) önbellek üret
python -m dronetrackingrl.real_data.experiment inspect-cache caches/....npz              # 3) dedektör kalitesini kontrol et
python -m dronetrackingrl.real_data.experiment train --train-cache ... --eval-cache ...  # 4) PPO eğit + raporla
```

`caches/`, `outputs/` ve `work/` büyük ve yeniden üretilebilir oldukları için
depoya alınmıyor (`.gitignore`).

## Durum (2026-09-21)

Kullanılabilir iki sahne var (dataset3: 6 kamera, dataset4: 7 kamera; dataset1/2'de
kameralar arası senkronizasyon tablosu, dataset5'te 2D etiket yok). Ajan rastgele
seçimden belirgin iyi, ama tek-sahne eğitiminde sahneler arası genelleme tutarsız —
bazı bölgelerde sabit kamerayı geçiyor, bir bölgede geçemiyor. **İki sahneyi birlikte
eğitmek bu başarısızlığı düzeltiyor.** Ayrıntı: [EXPERIMENTS.md](EXPERIMENTS.md).

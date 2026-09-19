# Başka PC'de koşu — kurulum ve çıktıyı geri getirme

Bu PC'de pahalı kısım (24GB'lık karelerden maskeleme) bir kez yapılıp küçük
önbellek dosyalarına (`caches/*.npz`) yazıldı. Diğer PC'de **veri setine
gerek yok**; sadece kod + bu önbellekler + Python yeterli. İnternet ve GPU
gerekmez (küçük model, CPU). Bellek ihtiyacı ~2-3GB.

## Pakette ne olmalı

```
DroneTransfer/
  code.zip                      # kod (git archive)
  caches/train_13442_23442.npz  # eğitim aralığı önbelleği
  caches/eval_28534_33875.npz   # bekletilmiş (held-out) test aralığı
  RUN_ON_OTHER_PC.md            # bu dosya
```

## 1) Kurulum (bir kez)

`code.zip`'i bir klasöre aç, o klasörde terminal aç. Python 3.10+ gerekir
(bu PC'de 3.12 ile test edildi).

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```
Linux / macOS:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e .
```
(`requirements-lock.txt` tam sürümleri sabitler. Bir paket senin işletim
sisteminde bulunamazsa `pip install -e .` bağımlılıkları kendisi çeker.)

İsteğe bağlı doğrulama (~1 dk): `pytest -q` → `50 passed` görmelisin.

Önbellekleri `caches/` klasörüne koy (repo klasörünün içine).

## 2) Önce hızlı gözlem-kalitesi kontrolü (saniyeler)

```
python -m dronetrackingrl.real_data.experiment inspect-cache caches/train_13442_23442.npz
```
Kamera başına maskelemenin drone'u ne kadar doğru bulduğunu (precision /
recall, piksel hatası) yazar. Maskeleme `SmallTargetDetector` ile yapıldı
(gerçek etiketlere karşı doğrulandı: ~%85 isabet, 2-4 piksel konum hatası);
eski yöntem (ilk kareye göre fark) gerçek çekimde çalışmadığı için terk edildi.

## 3) Koşuyu başlat

```
python -m dronetrackingrl.real_data.experiment train \
  --train-cache caches/train_13442_23442.npz \
  --eval-cache  caches/eval_28534_33875.npz \
  --out-dir outputs/run1 \
  --timesteps 100000 --seeds 0 1 2
```
(PowerShell'de satır sonu için `\` yerine `` ` `` kullan ya da tek satıra yaz.)

Ayarlar: `--timesteps` PPO adım sayısı, `--seeds` her seed ayrı bir eğitim
(3 seed = sonucun şansa bağlı olup olmadığını görmek için). Konsolda ilerleme
akar; aynı şey `outputs/run1/run.log`'a da yazılır.

## 4) Çıktıyı geri getir — TEK dosya

Koşu bitince:

```
outputs/run1.zip
```
oluşur (birkaç MB). Bunu USB / OneDrive / e-posta ile bu PC'ye getir ve yolunu
söylemen yeterli. İçinde:

| Dosya | Ne |
|---|---|
| `summary.txt` | İnsan okunur özet tablo (önce buna bak) |
| `results.json` | Tüm metrikler + gözlem tanıları + config |
| `run.log` | Koşu günlüğü |
| `seed_N/progress.csv` | PPO öğrenme eğrisi |
| `seed_N/trace_train.csv`, `trace_eval.csv` | Adım adım: hangi karede hangi kamera seçildi |
| `seed_N/ppo_model.zip`, `encoder.pt` | Eğitilmiş model ve encoder |

## Sonucu nasıl okumalı

`summary.txt`'te PPO'nun "görünür oranı"nı şunlarla karşılaştır:
- **Oracle**: ulaşılabilecek üst sınır
- **En iyi sabit kamera**: PPO buna yakınsa ajan kamera değiştirmeyi
  öğrenmemiş, tek kameraya yapışmış demektir
- **Rastgele**: alt sınır

Asıl güvenilir sayı `eval` (bekletilmiş aralık) satırıdır; `train` satırı
ezberi de içerebilir.

## Notlar

- Koşu yarıda kesilirse aynı komutu tekrar çalıştır (aynı `--out-dir`
  üzerine yazar).
- Önbellekleri yeniden üretmek istersen ham kareler gerekir (yalnızca ilk
  PC'de var): `python -m dronetrackingrl.real_data.experiment build-cache --help`

# Diğer PC'de eğitim — adım adım

Amaç: dataset3 ve dataset4 ile eğitim, ve **hiç görmediği sahnede** test
(dataset3'te eğit → dataset4'te test et, tersi, ve karışık).
Dataset 1/2'de kameralar arası senkronizasyon bilgisi, dataset 5'te 2D etiket
olmadığı için bunlar kullanılamıyor.

Sıra: **(1)** kurulum → **(2)** hızlı deneme → **(3)** kareleri çıkar →
**(4)** önbellek üret → **(5)** kontrol → **(6)** eğitim → **(7)** sonucu geri getir.

## Başlamadan önce

| Gereken | Ne kadar |
|---|---|
| Python | 3.10 veya üstü (bu PC'de 3.12 ile test edildi) |
| Boş disk | ~65 GB (dataset3 kareleri ~24GB, dataset4 ~30GB, geçici video birkaç GB) |
| RAM | ≥ 8 GB önerilir. Kare çıkarma / önbellek üretimi süreç başına ~0.5-1GB, eğitim ~5-6GB |
| Dataset | `drone-tracking-datasets` klasörü (indirdiğin), içinde `dataset3`, `dataset4` |

Aşağıdaki komutlarda iki yolu kendi PC'ne göre değiştir:
- `DATA` = `drone-tracking-datasets` klasörünün yolu
- `WORK` = kareler ve önbelleklerin yazılacağı, ~65GB boş yeri olan klasör

Komutları PowerShell'de tek satır olarak yaz (satır sonu `` ` ``); Linux/macOS'ta `\` kullanılır.

## 1) Kurulum (bir kez)

`code.zip`'i bir klasöre aç, o klasörde terminal aç:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```
Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate` sonra aynı iki `pip`.

Bir paket bulunamazsa `pip install -e .` bağımlılıkları kendisi çeker.
Doğrulama (~1 dk): `pytest -q` → **`68 passed`** görmelisin.

## 2) Hızlı deneme (1-2 dk) — kurulum çalışıyor mu?

Pakette iki küçük önbellek var (`caches/train_13442_23442.npz`,
`caches/eval_28534_33875.npz`; dataset3'ün bir kısmı). Bunlarla küçük bir koşu:

```powershell
python -m dronetrackingrl.real_data.experiment train --train-cache caches/train_13442_23442.npz --eval-cache caches/eval_28534_33875.npz --out-dir outputs/smoke --timesteps 3000 --encoder-epochs 1 --random-seeds 1 --ppo-n-steps 512
```
`outputs/smoke/summary.txt` oluşmalı ve hata olmamalı. (Sonuçlar anlamsız, sadece
kurulumu doğruluyor.)

## 3) Kareleri çıkar (dataset başına 10-25 dk)

Videolar `cam0.z01 … cam0.zip` gibi çok parçalı; standart araçlar açamıyor,
`prepare` aracı açar ve CRC32 ile doğrular.

```powershell
python -m dronetrackingrl.real_data.prepare extract --dataset-dir "DATA\dataset3" --frames-dir "WORK\dataset3_frames" --work-dir "WORK\tmp" --workers 3
python -m dronetrackingrl.real_data.prepare extract --dataset-dir "DATA\dataset4" --frames-dir "WORK\dataset4_frames" --work-dir "WORK\tmp" --workers 3
```
- `--workers` = aynı anda kaç kamera. RAM'e göre 2-5 seç (bellek yetmezse düşür).
- Sonunda **özet** çıkar; her kamera için `N / N kare [OK]` yazmalı:
  dataset3: 33875, 19960, 17166, 14196, 18900, 28080 kare.
  dataset4: 31075, 15409, 15678, 10933, 17640, 32016, 11292 kare.
- `UYUŞMAZLIK` yazarsa o kameranın videosu bozuk inmiş demektir; dataset'i yeniden indir.
- Yarıda kesilirse **aynı komutu tekrar çalıştır**, tamamlanmış kameraları atlar.

## 4) Önbellek üret (dataset başına ~15-45 dk, en uzun adım)

Kareleri okuyup maskeleme yapar ve küçük bir `.npz`'ye yazar. Kamera başına
ayrı süreç çalışır; `--parts-dir` sayesinde kesilirse kaldığı yerden devam eder.

```powershell
python -m dronetrackingrl.real_data.experiment build-cache --sync dataset3 --workers 4 --cameras 0 1 2 3 4 5 --frames-root "WORK\dataset3_frames" --detections-dir "DATA\dataset3\detections" --ref-start 1 --ref-end 33875 --parts-dir "WORK\parts3" --out caches/dataset3_full.npz

python -m dronetrackingrl.real_data.experiment build-cache --sync dataset4 --workers 4 --cameras 0 1 2 3 4 5 6 --frames-root "WORK\dataset4_frames" --detections-dir "DATA\dataset4\detections" --ref-start 1 --ref-end 31075 --parts-dir "WORK\parts4" --out caches/dataset4_full.npz
```
- Süre: bir kamera ~12 dk (kare hızı ~42/sn). `--workers 4` ile dataset3 ≈ 25 dk,
  dataset4 ≈ 25 dk. Süreç sayısı kamera sayısına yaklaştıkça hızlanır, ama her süreç
  ~1GB RAM ister.
- İlerleme satırları `[camN] önbellek: 5000/33875 kare (… kaldı)` şeklinde akar.
- **Bellek yetersizliğiyle kesilirse** (uygulama kapanır / "killed"): `--workers`'ı
  düşürüp **aynı komutu aynen tekrar çalıştır**. Biten kameralar (`parça kaydedildi`)
  yeniden yapılmaz.

## 5) Kontrol (saniyeler)

```powershell
python -m dronetrackingrl.real_data.experiment inspect-cache caches/dataset3_full.npz
python -m dronetrackingrl.real_data.experiment inspect-cache caches/dataset4_full.npz
```
Kamera başına şunlara bak: `precision` ≥ ~0.8 ve `centroid_error_px_median` birkaç
piksel (2-5). Bu, maskelemenin drone'u gerçekten bulduğunu gösterir. Bir kamerada
precision çok düşükse (≈ `gt_visible_rate`) veya hata yüzlerce pikselse bana yaz.

## 6) Eğitim

Ortak ayarlar (varsayılanlar zaten doğru): kameradan bağımsız politika
(`--policy shared`), gözlem standartlaştırma, `--ppo-gamma 0`. `--encoder-epochs 5`
büyük veride süreyi makul tutar. Her koşu 3 seed (`--seeds 0 1 2`).

**Deney A — sahneler arası (dataset3'te eğit, hiç görmediği dataset4'te test):**
```powershell
python -m dronetrackingrl.real_data.experiment train --train-cache caches/dataset3_full.npz --eval-cache caches/dataset4_full.npz --out-dir outputs/A_ds3_to_ds4 --timesteps 200000 --seeds 0 1 2 --encoder-epochs 5
```

**Deney B — ters yön (dataset4'te eğit, dataset3'te test):**
```powershell
python -m dronetrackingrl.real_data.experiment train --train-cache caches/dataset4_full.npz --eval-cache caches/dataset3_full.npz --out-dir outputs/B_ds4_to_ds3 --timesteps 200000 --seeds 0 1 2 --encoder-epochs 5
```

**Deney C — karışık (ikisinin de ilk %80'i eğitim, son %20'si ayrı ayrı test):**
```powershell
python -m dronetrackingrl.real_data.experiment train --train-cache caches/dataset3_full.npz --train-range 1 27100 --extra-train caches/dataset4_full.npz:1:24860 --eval-cache caches/dataset3_full.npz --eval-range 27101 33875 --extra-eval caches/dataset4_full.npz:24861:31075 --out-dir outputs/C_mixed --timesteps 200000 --seeds 0 1 2 --encoder-epochs 5
```
- Sırayla çalıştır (A, sonra B, sonra C); her biri kendi klasörüne yazar.
- Konsol ve `outputs/<koşu>/run.log` ilerlemeyi gösterir. Süre veri boyutuna
  bağlı: encoder ön-eğitimi (seed başına) en uzun kısım olabilir; ilk seed'in
  `PPO eğitimi` satırına geçmesini bekle.
- Bellek yetmezse: `--encoder-batch-size 128` ve/veya seed'leri ayrı koşularda çalıştır
  (`--seeds 0`, sonra `--seeds 1` …; farklı `--out-dir`).

## 7) Sonucu geri getir

Her koşu bitince `outputs/<koşu>.zip` oluşur (birkaç MB). Üç koşunun zip'ini
USB / OneDrive / e-posta ile getir, yolunu söyle. İçinde `summary.txt` (özet tablo;
önce buna bak), `results.json`, `run.log`, seed başına öğrenme eğrisi, adım adım kamera
seçim kayıtları ve eğitilmiş model var.

**Sonuç nasıl okunur:** `summary.txt`'te PPO'nun "görünür oranı"nı **Oracle** (üst
sınır), **en iyi sabit kamera** ve **rastgele** ile karşılaştır. PPO oracle'a yakın ve
sabit kameradan iyiyse ajan gözlemi kullanarak kamera seçiyor demektir. Asıl güvenilir
satırlar `eval` (hiç görmediği veri) satırlarıdır, `train` ezberi de içerir.

# RELO hibrit dedektör — tasarım (2026-10-08)

## Amaç
Kamera başına drone konumunu `SmallTargetDetector` yerine (onunla birlikte) RELO
takipçisiyle bulmak; RL ajanı kamera seçimini bu daha iyi gözlemle yapsın.
Kaynak: RELO (Chen vd., ICML 2026, arXiv 2605.07379), açık ağırlıklar HF `xche32/RELO`.

**Başarı ölçütü (iki kapı):**
1. *Dedektör düzeyi:* aynı pencerelerde `observation_diagnostics` (precision, recall,
   piksel hatası) hibritte SmallTargetDetector'dan daha iyi — özellikle ds4 cam0
   (recall) ve cam6 (precision). Daha iyi değilse burada durulur ve raporlanır.
2. *RL düzeyi:* C_hist ayarlarıyla (latent+aux, own-history, switch_penalty 0.1,
   3 tohum) `latent+aux+relo`; aynı held-out bölgelerde + fold B'de `visible_rate`.

**Değişmeyenler:** ödül yalnızca ground truth'tan; politika/PPO/encoder aynı;
önbellek tabanlı iki aşamalı akış aynı.

## Bileşenler
- `masking/relo_tracker.py` — `ReloTracker(relo_root, variant="t256", device=None)`:
  RELO deposunu `sys.path`'e ekleyip *tembel* içe aktarır (testler RELO'suz çalışır).
  `init(frame_bgr, box_xywh)`, `track(frame_bgr) -> (box_xywh, confidence)`.
  Güven = politika logit haritasının (16×16, Hann penceresi uygulanmadan) softmax
  maksimumu, [0,1]. Kutu en az `min_box_px` olacak şekilde kırpılır (minik hedefte
  çökmesin). Şablon güncelleme kapalı (RELO varsayılanı, T256 tek şablon).
- `masking/hybrid_detector.py` — `HybridReloDetector(crop_size, tracker_factory, ...)`,
  arayüzü SmallTargetDetector ile aynı (`reset()`, `update(frame) -> MaskResult`).
  Klasik dedektör her karede çalışır (arka plan modeli kesintisiz). İki durum:
  - **ARAMA:** çıktı = klasik dedektörün çıktısı. Klasik dedektör `init_streak`
    ardışık karede (her biri öncekine `persist_radius` içinde) tespit verirse RELO,
    o merkezde `init_box_px` kare kutuyla başlatılır → TAKİP.
  - **TAKİP:** çıktı merkezi = RELO kutu merkezi. Güven `lost_patience` kare
    üst üste `lost_threshold` altındaysa → ARAMA (o karede çıktı klasik dedektörün).
  - Kırpım: klasik dedektörün blackhat tepkisinden, çıktı merkezi etrafında
    (encoder'ın gördüğü sinyal türü değişmez).
- `MaskResult`'a varsayılanlı alanlar: `relo_score: Optional[float]`,
  `relo_box_wh: Optional[Tuple[float,float]]`, `relo_tracking: bool=False`.
- `SmallTargetDetector`: son blackhat tepkisini ve ölçeği dışarı verir
  (`last_response`, `last_scale`) — davranış değişmez.
- `RealCameraSignalProvider`: `last_mask_results` (kamera başına son MaskResult)
  ve renkli kare okuyucu seçeneği.
- **Önbellek v3:** yeni diziler `relo_score` (float32, NaN), `relo_box_wh`
  (float32, NaN), `relo_tracking` (bool). v2 dosyaları yüklenir (alanlar NaN/False).
  `slice`/`pad_cameras`/`merge_camera_caches` yeni alanları taşır.
- **Gözlem modu `latent+aux+relo`:** aux'a 4 özellik: takip bayrağı, güven (yoksa 0),
  kutu w ve h (/100 px, yoksa 0). `RELO_FEATURE_DIM = 4`.
- **CLI:** `build-cache --detector {classic,relo} --relo-root --relo-variant
  --relo-init-box --relo-lost-threshold --relo-lost-patience --relo-init-streak`;
  relo ile paralel üretim `spawn` bağlamında (CUDA + fork güvenli değil).

## Ayar
`init_box_px`, `lost_threshold`, `lost_patience`, `init_streak` dataset3'ün ayar
pencerelerinde küçük bir ızgarayla seçilir; RL held-out/test bölgelerine dokunulmaz.
Varsayılanlar ilk deneme için: init_box 20 px, lost_threshold 0.2, lost_patience 5,
init_streak 3 (ayar sonrası güncellenir ve EXPERIMENTS.md'ye yazılır).

## Ortam
5060 masaüstü (Blackwell): proje venv'i torch CPU. RELO için ayrı venv
(`.venv-relo`, CUDA'lı torch cu128 + RELO gereksinimleri + bu paket `-e`).
RELO deposu ve ağırlıkları `/media/noron/DISK02/DroneRL/RELO`.

## Test
- Sahte takipçiyle: ARAMA→TAKİP→ARAMA geçişleri, kırpım merkezi, MaskResult alanları.
- Önbellek: v3 kaydet/yükle, v2 yükleme geriye uyumu, slice/pad/merge alanları.
- Gözlem: `latent+aux+relo` boyutu ve özellik değerleri.
- Gerçek RELO duman testi: RELO kurulu değilse `skip`.

## Riskler
- Drone çok küçük: arama bölgesi (4×kutu → 256 px) aşırı büyütme; init_box ayarı kritik.
- Başlatma klasik dedektöre bağlı: cam0'ın kaçırmaları ARAMA'da sürer.
- RELO deposunda lisans dosyası yok — yalnızca araştırma içi kullanım, kodu
  bu repoya kopyalamıyoruz.

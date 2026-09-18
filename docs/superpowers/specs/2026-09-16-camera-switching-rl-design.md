# Çoklu Kamera RL ile Drone Takibi — Tasarım Spec'i

Tarih: 2026-09-16 (ilk yazım) — 2026-09-18 güncelleme: veri seti indirmesi
tamamlandı, gerçek veri yapısı incelendi. "Genel Veri Akışı", "Bileşen 3"
ve "Baseline'lar ve Değerlendirme" bölümleri gerçek veriye göre revize
edildi (bkz. § Gerçek Veri Entegrasyonu). Eski "Açık Noktalar" bölümü
kaldırıldı, yerini gerçek bulgular aldı.

Durum: Onaylandı (brainstorming aşaması tamamlandı). Faz 1'in
veri-setinden-bağımsız çekirdek bileşenleri (aşağıdaki Bileşen 1-3'ün ilk
sürümü) zaten uygulandı ve `master`'a merge edildi — bkz.
`docs/superpowers/plans/2026-09-16-core-rl-components.md`. Bu güncelleme,
aynı Faz 1 kapsamında, gerçek veriyle çalışacak ikinci uygulama turunun
(Faz 1b) tasarımını ekliyor; Faz 1'in kendisini değiştirmiyor, üzerine inşa
ediyor.

## Amaç ve Kapsam

Farklı açılardan aynı sahneyi izleyen birden fazla sabit kamera, bir drone'u
kayıt altına alıyor. Amaç, bir RL ajanının her an "hangi kamera drone'u en
iyi gösteriyorsa" ona geçiş yapmasını öğrenmesi; böylece drone hiçbir zaman
kaybedilmeden kesintisiz bir 2D yörünge (trajectory) çıkarılacak (Faz 1).
Faz 1 başarılı olursa çoklu kamera görüşlerinden 3D yörünge rekonstrüksiyonuna
geçilecek (Faz 2).

Nihai hedef sırası: önce çalışan bir sistem ortaya koymak, ardından bu
konudaki literatür boşluğunu dolduran bir makale/bildiri yazmak. Bu nedenle
baseline karşılaştırmaları ve ölçülebilir metrikler baştan tasarıma dahil
edildi.

## Veri Seti

Kaynak: [CenekAlbl/drone-tracking-datasets](https://github.com/CenekAlbl/drone-tracking-datasets)
(ETH Zürich). İndirildi: `C:\Users\BalBa\OneDrive\Desktop\DatasetDroneTracker\drone-tracking-datasets\`.

5 ayrı sahne (`dataset1`..`dataset5`), her biri kendi klasöründe:

| Dataset | Kamera | Süre | 3D yörünge | Senkronizasyon | Kamera konumu | 2D etiket |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | ~2 dk | Var | Yok | Yok | Var |
| 2 | 4 | ~2.5 dk | Var | Yok | Yok | Var |
| 3 | 6 | ~9 dk | Var | Var | Var | Var |
| 4 | 7 | ~7 dk | Var | Var | Yok | Var |
| 5 | 6, 3 drone | ~10 dk | Var | Var | Var | Yok |

Her dataset klasöründe:
- `camN/camN.zip` (+ bazılarında çok parçalı `camN.z01`, `camN.z02`, ...) —
  o kameranın ham videosu (mp4). **Kareye bölünmemiş**, sıkıştırılmış video
  olarak duruyor; kare çıkarma (frame extraction) bizim yapmamız gerekiyor
  (OpenCV `cv2.VideoCapture`, sistemde ffmpeg binary'si yok).
- `cameras.txt` — kamera id → model eşlemesi (örn. `cam0 - iphone6`).
  Model adı `calibration/<model>/<model>.json` dosyasına işaret eder
  (K-matrix, distCoeff, fps, çözünürlük — intrinsic kalibrasyon, tüm
  dataset'ler arasında paylaşılan, cihaz modeline göre).
- `detections/camN.txt` — **elle etiketlenmiş, hazır 2D ground truth**.
  Format: `frame_id x y` (1 satır = 1 kare), **sadece drone'un o kamerada
  görünür/ayırt edilebilir olduğu kareler** için satır var. Yani "kadrajda
  mı" + piksel konumu zaten hazır; XYZ+kalibrasyon projeksiyonuna gerek yok.
- `trajectory/rtk.txt` — drone'un 3D (X,Y,Z) ground-truth yörüngesi (RTK ile
  ölçülmüş), satır sırası kendi örnekleme hızında. **Kameralarla senkronize
  değil ve senkronizasyon kuralı hiçbir yerde belgelenmemiş** (dataset3
  rtk.txt 3306 satır, dataset1 rtk.txt 3291 satır — tamamen farklı uçuş
  sürelerine rağmen neredeyse aynı satır sayısı, satır↔kare eşlemesi yok).
  Bu yüzden Faz 1b'de kullanılmıyor — bkz. § Gerçek Veri Entegrasyonu.
- (sadece dataset3, dataset5) `camera-locations/campos.txt` — her kameranın
  gerçek 3D (X,Y,Z) **konumu** (rotation/yönelim verilmiyor). Tek başına
  tam projeksiyon/triangülasyon için yetersiz (kamera yönelimi olmadan);
  Faz 1b'de kullanılmıyor, Faz 2 (3D) için ileride kamera pozu tahmini
  (structure-from-motion) gerektirecek.
- (sadece dataset3, dataset4, dataset5) dataset'in kendi `README.md`'sinde
  kameralar arası **gerçek zaman senkronizasyon parametreleri** (alpha/beta,
  bkz. § Gerçek Veri Entegrasyonu).

**Önemli bulgu:** incelenen dataset'lerde (1 ve 3) her kameranın kendi
klibinde drone **%100 kare boyunca görünür** — occlusion/kaybetme senaryosu
videoların içinde yok (klipler zaten sadece drone görünürken
kaydedilmiş/kırpılmış). Kamera-değiştirme sinyali occlusion'dan değil,
kameraların **farklı zamanlarda başlayıp bitmesinden** geliyor — bkz.
§ Gerçek Veri Entegrasyonu.

## Genel Veri Akışı

İki ayrı, birbirine karıştırılmayan kullanım (sentetik-veri Faz 1'de olduğu
gibi):

1. **Maskeleme → RL'nin gerçekte gördüğü sinyal.** Kameralar sabit olduğu
   için arka plan çıkarma (background subtraction / frame differencing) ile
   drone arka plandan ayrılır; tamamen görüntü tabanlı, kör çalışır (ground
   truth'a bakmaz).
2. **Ground truth → sadece otomatik doğrulama/reward referansı.** Sentetik
   Faz 1'de bu XYZ+kalibrasyon projeksiyonundan geliyordu. **Gerçek veride
   yerini `detections/camN.txt`'teki hazır elle-etiketlenmiş 2D konumlar
   alıyor** — ayrıca projeksiyona gerek yok, çünkü etiket zaten piksel
   cinsinden var. XYZ+kalibrasyon projeksiyon modülü (`geometry/projection.py`,
   Faz 1'in Task 1'i) kapsamdan çıkmadı; sadece rolü değişti — artık asıl
   ground-truth kaynağı değil, ileride Faz 2 (3D) için saklı tutuluyor (bkz.
   § Gerçek Veri Entegrasyonu — `campos.txt` kamera yönelimi eksikliği
   nedeniyle Faz 1b'de mesafe hesabı için de kullanılmıyor).

## Bileşen 1: Maskeleme (Algı) Modülü

- Her kamera akışı için bağımsız çalışır (kameralar arası bilgi paylaşımı yok).
- Yöntem: arka plan çıkarma / frame differencing (küçük hareketli cisim +
  sabit kamera için standart, eğitim verisi gerektirmeyen klasik CV yöntemi).
- Çıktı (her kare, her kamera için), iki ayrı kullanım için:
  1. **Normalize edilmiş maskelenmiş kırpım** (sabit boyuta resize, örn.
     64×64, arka plan karartılmış/grayscale) → RL gözlemi için (bkz.
     Bileşen 2).
  2. **Blob merkez pikseli** (varsa) → **sadece nihai trajectory kaydı
     için**, RL'nin gözlemine hiç dahil edilmez.
  3. `görünür_mü (bool)` — blob bulunamazsa `False`.
- Doğrulama: blob merkezi ile XYZ+kalibrasyondan projekte edilen "gerçek
  piksel konumu" karşılaştırılarak maskeleme modülünün doğruluğu (piksel
  hatası, kaçırma oranı) otomatik ölçülür.

## Bileşen 2: Latent Encoder

- Kameraların modelleri farklı ve birden fazla drone tipi olduğu için
  **tüm kameralardan/drone tiplerinden gelen maskelenmiş kırpımlar
  havuzlanarak tek, paylaşılan bir autoencoder eğitilir** (kamera başına
  ayrı model yok — daha fazla veri, daha genellenebilir temsil, tek model
  yönetimi).
- Girdi: Bileşen 1'in ürettiği normalize maskelenmiş kırpım.
- Çıktı: sabit boyutlu latent vektör.
- Bu encoder RL eğitiminden **önce**, unsupervised olarak ayrıca eğitilir.

## Bileşen 3: RL Ortamı (Gymnasium + Stable-Baselines3, PPO)

- **Episode**: bir video sekansı, her adım bir kare.
- **Gözlem (observation)**: o karede N kameranın her birinden gelen latent
  vektörlerin birleşimi (concatenation). Elle çıkarılan (x/y/confidence)
  sinyali gözleme dahil edilmez — latent onun yerine geçer.
- **Aksiyon (action)**: `Discrete(N)` — o kare için hangi kameraya
  geçileceği.
- **Ödül (reward) — Faz 1 (sentetik, uygulandı)**: seçilen kameranın
  ground-truth görünürlüğü (`görünür_mü`) — kadrajdaysa +1, değilse -1.
  Sadece "kadrajda olma" kriteri; bu basit tasarım sentetik doğrulama için
  bilerek korundu ve değiştirilmedi (bkz. `CameraSwitchEnv` implementasyonu).
- **Ödül (reward) — Faz 1b (gerçek veri, bu güncellemede tasarlandı)**:
  gerçek veride her kameranın kendi klibinde drone sürekli görünür olduğu
  için (bkz. § Veri Seti bulgusu), salt ikili "görünür mü" sinyali neredeyse
  sabit kalır ve öğrenilecek bir şey bırakmaz. Bunun yerine **sürekli bir
  kalite skoru** kullanılıyor — bkz. § Gerçek Veri Entegrasyonu için tam
  formül. Kamera o an senkronize zaman diliminde kayıtta değilse kalite=0
  (görünmüyor sayılır).
- **Algoritma**: PPO (discrete aksiyon uzayına doğrudan uygun,
  Stable-Baselines3 üzerinden).
- Farklı sekanslarda kamera sayısı (N) farklıysa, her N konfigürasyonu ayrı
  bir ortam/model olarak ele alınır (gözlem vektör boyutu sabit tutulmalı).

## Nihai Çıktı: 2D Trajectory

Her karede seçilen kameranın blob merkez pikseli (Bileşen 1'in yan ürünü)
zaman serisi halinde birleştirilir → **Faz 1'in nihai 2D trajectory çıktısı**.
Kamera seçim kaydı (hangi karede hangi kamera seçildi) ayrıca loglanır.

## Baseline'lar ve Değerlendirme

Karşılaştırma seti:
- **Oracle**: her karede gerçekten en yüksek kalite skoruna (Faz 1: kadrajda
  olma; Faz 1b: yakınlık+merkezleme) sahip kamerayı seçen üst sınır.
- **Sabit tek kamera**: sekans başına en iyi ortalama performans veren tek
  kamerada kalmak.
- **Rastgele/round-robin geçiş**: alt sınır.
- **PPO ajanı**: önerilen yöntem.

Metrikler:
- Faz 1: kare başına "drone kadrajda mı" oranı. Faz 1b: kare başına
  ortalama kalite skoru (RL vs oracle vs baseline'lar).
- Piksel trajectory hatası: RL'nin çıkardığı 2D yörünge vs ground-truth
  (Faz 1: XYZ projeksiyonu; Faz 1b: `detections/camN.txt`).
- Gereksiz kamera geçiş sayısı (flickering göstergesi, ödüle dahil değil
  ama raporlanacak).

## Hata / Uç Durumlar

- Hiçbir kamerada drone görünmüyorsa: ajan yine de bir kamera seçmek zorunda
  (aksiyon uzayı boş bırakılamaz), o adım için ödül negatif/0, episode
  devam eder.
- Sekanslar arası kamera sayısı (N) farklılığı: sabit N varsayımıyla
  sekanslar N'e göre gruplanır; farklı N'li sekanslar ayrı ortam
  konfigürasyonu olarak ele alınır.
- Bozuk/eksik kare dosyaları: pipeline atlar + loglar, sessizce yutmaz.

## Test / Doğrulama Planı

- **Maskeleme modülü**: örnek kareler üzerinde görsel + ground-truth
  projeksiyonla sayısal karşılaştırma (kaçırma oranı, piksel hatası).
- **RL ortamı**: rastgele politika ile sanity check (reward'lar beklenen
  aralıkta mı), oracle politika ile üst sınırın doğrulanması.
- **Eğitim**: reward eğrisinin baseline'ları (sabit kamera, rastgele) geçtiğinin
  gösterilmesi.

## Faz 2 (kapsam dışı, ileride)

Faz 1 başarılı olursa: eşzamanlı olarak "iyi" (kadrajda) durumdaki birden
fazla kamera görüşünü kalibrasyon verisiyle triangüle ederek 3D trajectory
rekonstrüksiyonuna geçilecek. Bu spec'in kapsamı dışında; ayrı bir
brainstorming/spec döngüsü ile ele alınacak.

## Gerçek Veri Entegrasyonu (Faz 1b) — 2026-09-18 eklendi

Faz 1'in çekirdek bileşenleri (Bileşen 1-3, önceki sürüm) sentetik veriyle
inşa edildi ve doğrulandı. Bu bölüm, aynı bileşenleri gerçek
`drone-tracking-datasets` verisine bağlayan ikinci uygulama turunun
tasarımıdır — ayrı bir plan dosyasında (`docs/superpowers/plans/`,
tarih 2026-09-18) TDD task'larına dökülecek.

### Dataset seçimi

- **dataset1**: sadece boru hattı (pipeline) doğrulaması için — kareler
  zaten çıkarıldı (4 kamera, 22.371 kare, 4.4GB). Kamera-değiştirme sinyali
  yok (her kamerada %100 görünür), bu yüzden **asıl eğitim için kullanılmaz**.
- **dataset3**: asıl eğitim/değerlendirme hedefi. Senkronizasyon
  parametreleri + gerçek kamera konumları (campos.txt) olan, kamera sayısı
  ve süre açısından da orta zorlukta tek dataset. Kareler çıkarılıyor
  (6 kamera, ~132.000 kare, tahmini ~25GB).
- dataset2/4/5 bu turun kapsamı dışında; ileride aynı adaptörlerle
  eklenebilir (dataset4 senkronizasyon var ama kamera konumu yok — sadece
  merkezleme bileşeniyle çalışır; dataset5 çoklu drone + 2D etiket yok, ayrı
  bir tasarım gerektirir).

### Senkronize zaman ekseni

Kameralar farklı zamanlarda başlayıp bitiyor (farklı kare sayıları: dataset3
için 14196'dan 33875'e kadar) ve farklı, bazen değişken fps'lerde kayıt
yapıyor. dataset3'ün README'si her kamera çifti için gerçek ölçülmüş
zaman-eşleme parametreleri (`alpha`, `beta`) veriyor:

```
frame_j = alpha(i→j) * frame_i + beta(i→j)
```

Referans kamera olarak **cam0** seçildi (en yüksek fps=59.94, en uzun kayıt).
Her kamera N için, cam0'ın frame_id ekseni boyunca karşılık gelen `frame_j`
hesaplanır; bu değer o kameranın `detections/camN.txt`'indeki geçerli
aralığın (0..max_frame_id) **dışına düşerse**, o kamera o an "kayıt dışı" =
görünmüyor sayılır. Gerçek sayılarla doğrulandı: dataset3'te cam3 ve cam5,
cam0'ın kaydının son ~%1-1.4'ünde bu şekilde devre dışı kalıyor — RL'nin
öğreneceği kamera-değiştirme sinyali tam olarak burada.

### Kalite skoru (yeni reward tanımı)

```
kalite = 0.5 * yakınlık_norm + 0.5 * merkezleme_norm
```

**Düzeltme (2026-09-18):** İlk tasarım `campos.txt`+`rtk.txt` ile gerçek 3D
mesafe hesaplamayı öngörüyordu; bu terk edildi çünkü (a) `rtk.txt`'nin
kameralarla senkronizasyon kuralı belgelenmemiş, (b) kamera yönelimi
(rotation) verisi yok. Yerine, ekstra veri gerektirmeyen görüntü-tabanlı bir
vekil (proxy) kullanılıyor:

- **yakınlık_norm**: `BackgroundSubtractor`'ın zaten hesapladığı **blob
  alanı** (piksel², connected-components'tan `stats[..., CC_STAT_AREA]`) —
  Bileşen 1'in `MaskResult`'ına yeni bir `blob_area: Optional[int]` alanı
  eklenecek (mevcut `mask_crop`/`centroid`/`visible` alanlarına ek, geriye
  dönük uyumlu). **Kamera başına** min-max normalize edilir (o kameranın
  gördüğü en büyük blob = 1, en küçük/yok = 0) — drone kameraya ne kadar
  yakınsa görüntüde o kadar büyük görünür, mutlak piksel alanı kameralar
  arasında karşılaştırılabilir değil (çözünürlük/mesafe farklı).
- **merkezleme_norm**: `detections/camN.txt`'teki (x,y) konumunun görüntü
  merkezine piksel uzaklığı, o kameranın çözünürlüğünün (kalibrasyon
  json'undan) yarı-köşegenine göre normalize edilir (merkez = 1, kenar/dışı
  ≈ 0, clamp edilir).
- Kamera senkronize zaman diliminde kayıt dışıysa: kalite = 0.
- Ağırlıklar (0.5/0.5) onaylandı; ileride ampirik olarak ayarlanabilir.
- `campos.txt`/`rtk.txt` bu turda kullanılmıyor (yukarıya bkz.) — Faz 2 (3D)
  için saklı tutuluyor.

### Gerekli yeni parser'lar / bileşenler (Faz 1b'nin kapsamı)

- `cameras.txt` parser → kamera id → model adı.
- Kalibrasyon `<model>.json` parser → K-matrix, çözünürlük (merkezleme
  normalizasyonu için).
- `detections/camN.txt` parser → sparse `{frame_id: (x, y)}` haritası.
- `MaskResult`'a `blob_area` alanı eklenmesi (Bileşen 1'in küçük, geriye
  dönük uyumlu bir uzantısı — `background_subtraction.py`'de zaten
  hesaplanan `stats[..., CC_STAT_AREA]` değerinin döndürülen sonuca
  eklenmesi).
- Senkronize zaman ekseni oluşturucu (alpha/beta tablosu → her cam0 karesi
  için diğer kameraların karşılık gelen frame_id'si + kayıtta-mı durumu).
- Kalite skoru hesaplayıcı (yukarıdaki formül).
- `RealCameraSignalProvider` — Faz 1'in `CameraSignalProvider` protokolünü
  gerçek karelerle (disk'ten okunan `.jpg`) + gerçek kalite skoruyla
  dolduran implementasyon; mevcut `BackgroundSubtractor` ve paylaşılan
  `MaskedCropAutoencoder`'ı olduğu gibi kullanır (bu ikisi Faz 1'de zaten
  yazıldı, değişmiyor).
- Çok parçalı zip'lerden video çıkarma script'i zaten yazıldı ve doğrulandı:
  `DatasetDroneTracker/drone-tracking-datasets/_tools/extract_multipart_zip.py`.

### Açık noktalar (bu turun kapsamı dışında, ileride)

- Mate 7 gibi bazı telefonların değişken fps kaydettiği README'de not
  düşülmüş (ffprobe ile per-frame timestamp çıkarma öneriliyor) — dataset3
  için alpha/beta tablosu bu sorunu zaten 30fps'e remap ederek çözmüş
  durumda, bu yüzden bu turda ekstra işlem gerekmiyor; dataset4/5'e
  geçildiğinde tekrar değerlendirilmeli.
- Distorsiyon katsayıları (`distCoeff`) şu an kullanılmıyor (K-matrix ile
  düz pinhole varsayılıyor) — merkezleme skorunun hassasiyetini önemli
  ölçüde etkiliyorsa ileride eklenebilir.

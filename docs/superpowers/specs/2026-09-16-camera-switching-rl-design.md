# Çoklu Kamera RL ile Drone Takibi — Tasarım Spec'i

Tarih: 2026-09-16
Durum: Onaylandı (brainstorming aşaması tamamlandı)

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
(indirme tamamlandığında burada netleştirilecek: sekans sayısı, her sekansta
kaç kamera, kare sayısı/fps, dosya adlandırma kuralı).

Bilinen içerik:
- Her kamera için model/kalibrasyon bilgisi (intrinsic/extrinsic — "model").
- Drone'un dünya koordinatındaki (X, Y, Z) ground-truth yörüngesi.
- Videodan kareye bölünmüş, **etiketlenmemiş** ham görüntüler (her kamera için).

## Genel Veri Akışı

İki ayrı, birbirine karıştırılmayan kullanım:

1. **Maskeleme → RL'nin gerçekte gördüğü sinyal.** Kameralar sabit olduğu
   için arka plan çıkarma (background subtraction / frame differencing) ile
   drone arka plandan ayrılır; tamamen görüntü tabanlı, kör çalışır (XYZ
   ground-truth'a bakmaz).
2. **XYZ + kalibrasyon → sadece otomatik doğrulama/ground-truth referansı.**
   XYZ, her kameranın izdüşüm matrisiyle 2D'ye projekte edilerek "gerçekte
   drone bu pikselde olmalıydı" referansı otomatik üretilir — elle etiketleme
   yok. Bu referans (a) maskeleme modülünün doğruluğunu ölçmek, (b) RL
   ödülünü hesaplamak, (c) nihai trajectory'nin doğruluğunu değerlendirmek
   için kullanılır.

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
- **Ödül (reward)**: seçilen kameranın XYZ+kalibrasyondan projekte edilen
  ground-truth görünürlüğü (`görünür_mü`) — kadrajdaysa pozitif, değilse
  negatif/0. (Reward kriteri olarak sadece "kadrajda olma/görüntülenebilirlik"
  seçildi; merkeze yakınlık, boyut, geçiş cezası gibi ek kriterler bu
  aşamada dahil edilmedi.)
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
- **Oracle**: her karede ground-truth projeksiyonuna göre gerçekten en iyi
  (kadrajda) kamerayı seçen üst sınır.
- **Sabit tek kamera**: sekans başına en iyi ortalama performans veren tek
  kamerada kalmak.
- **Rastgele/round-robin geçiş**: alt sınır.
- **PPO ajanı**: önerilen yöntem.

Metrikler:
- Kare başına "drone kadrajda mı" oranı (RL vs oracle vs baseline'lar).
- Piksel trajectory hatası: RL'nin çıkardığı 2D yörünge vs ground-truth
  projeksiyonu.
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

## Açık Noktalar (veri seti indirmesi tamamlanınca netleşecek)

- Sekans sayısı, sekans başına kamera sayısı (N), kare sayısı/fps.
- Kalibrasyon dosyalarının tam formatı (intrinsic/extrinsic parametreleri
  nasıl saklanmış).
- XYZ ground-truth dosyasının zaman senkronizasyonu (kameralarla aynı kare
  indeksine mi hizalı, yoksa ayrı bir zaman damgası mı var).

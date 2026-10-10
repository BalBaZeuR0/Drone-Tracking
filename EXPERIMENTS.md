# Deney günlüğü

Bu dosya, tüm gerçek-veri deneylerinin sonucunu — başarılı olsun olmasın — kaydeder.
Amaç: geçersiz/başarısız denemeleri de görünür tutmak, aynı hataya tekrar düşmemek ve
"neden bu tasarımı seçtik" sorusuna geriye dönük cevap verebilmek. Deneyler
`python -m dronetrackingrl.real_data.experiment train ...` ile, sadece önbellekten
(`caches/*.npz`) çalışır; ölçüt her yerde **görünür oranı** (ajanın seçtiği kamerada
drone'un o karede gerçekten görünme yüzdesi, %). Ayrıntılı yöntem/okuma rehberi için
[Rapor (2026-09-21)](https://claude.ai/artifact/3m6Zoe1PMFg2cngrH6ky9W).

Kısaltmalar: **sabit** = en iyi sabit kamera (test verisine bakılarak seçilir, ajana
karşı iyimser bir ölçüt) · **oracle** = etiketleri bilen hayali ajan, aşılamayan üst
sınır · `latent` = sadece encoder çıktısı · `latent+aux` = + 5 nedensel dedektör
özelliği (iz uzunluğu, ateşleme oranı, …) · `gt` = gözlem = etiket (sadece ablasyon).

## Özet tablo

| # | Deney | Gözlem | Eğitim | Test | Sonuç (PPO vs sabit) | Durum |
|---|---|---|---|---|---|---|
| A | ds3 → ds4 | latent | dataset3 tümü | dataset4 1–26000 (görülmemiş) | 85,3 vs 87,8 | ❌ **sabiti geçemedi** |
| B | ds4 → ds3 | latent | dataset4 1–26000 | dataset3 tümü (görülmemiş) | 94,0 vs 94,1 | ⚠️ **eşit** |
| D | ds4 blok bölünmesi (ilk deneme) | latent | ds4 1–4800 + 10201–15800 + 18201–26000 | ds4 24861–31075 | oracle = sabit = 29,9 | 🚫 **geçersiz** — test bölümünde drone hiç yoktu, ölçüm anlamsız |
| D2 | ds4 blok bölünmesi (düzeltilmiş) | latent | aynı (D ile) | ds4 5001–10000 ve 16001–18000 | 76,3 vs 76,5 · 95,8 vs 90,2 | ⚠️/✅ karışık |
| G | gt ablasyonu | **gt** | dataset3 | dataset4 1–26000 | 99,1 = oracle | ✅ **sağlamlık kanıtlandı** (RL düzeneği sorunlu değil) |
| A2 | ds3 → ds4 | latent+aux | dataset3 tümü | dataset4 1–26000 (görülmemiş) | 87,9 vs 87,8 | ⚠️ **eşit** (fark tohum sapması ±0,9 içinde) |
| B2 | ds4 → ds3 | latent+aux | dataset4 1–26000 | dataset3 tümü (görülmemiş) | 94,4 vs 94,1 | ⚠️ **eşit** |
| D3 | ds4 blok bölünmesi | latent+aux | aynı (D2 ile) | ds4 5001–10000 ve 16001–18000 | 85,1 vs 76,5 · 96,6 vs 90,2 | ✅ **sabiti geçti** |
| C | karışık (ds3+ds4) | latent+aux | ds3+ds4'ün ~%80'i (4 blok halinde) | 4 ortak held-out bölge | aşağıda | ✅ **en tutarlı** |
| C3 | sadece ds3 | latent+aux | sadece dataset3 | aynı 4 bölge | aşağıda | ⚠️ **cross-scene'de bir bölgede başarısız** |
| C4 | sadece ds4 | latent+aux | sadece dataset4 | aynı 4 bölge | aşağıda | ✅ iyi |
| C_hist | karışık + kendi geçmişi + geçiş cezası | latent+aux | ds3+ds4 (C ile aynı) | aynı 4 bölge | aşağıda | ✅ **geçiş sayısı 28-118× azaldı, doğruluk aynı/daha iyi** |

## C / C3 / C4 — karışık eğitim gerçekten yardımcı oluyor mu? (2026-09-21, 3 tohum)

Aynı 4 held-out bölgede (eğitimde hiç kullanılmadı, en yakın eğitim ucuna ~200 kare /
~7 sn tampon var), üç farklı eğitim verisiyle test:

| Test bölgesi | Sabit | **C** (ds3+ds4) | C3 (sadece ds3) | C4 (sadece ds4) | Oracle |
|---|---|---|---|---|---|
| ds3 12001–14000 (aynı sahne, C3 için) | 83,7 | 86,7 | 86,8 | 86,2 *(cross-scene)* | 94,7 |
| ds3 28001–30000 (aynı sahne, C3 için) | 65,8 | 99,5 | 99,9 | 99,4 *(cross-scene)* | 100,0 |
| ds4 5001–10000 (aynı sahne, C4 için) | 76,5 | 85,4 | 83,8 *(cross-scene)* | 85,1 | 95,1 |
| ds4 16001–18000 (aynı sahne, C4 için) | 90,2 | 94,8 | **84,1** *(cross-scene)* | 96,6 | 100,0 |

**Bulgu:** cross-scene aktarım tutarlı değil — 3/4 bölgede sabit kamerayı geçiyor
(bazen büyük farkla, ds3 28001–30000'de +33,6 puan), ama **1/4 bölgede sabit kameranın
altına düşüyor** (C3, ds4 16001–18000'de 84,1 vs 90,2 — tek-sahne eğitimin
genellemediği somut bir örnek). **Karışık eğitim (C) bu başarısızlığı büyük ölçüde
düzeltiyor** (84,1 → 94,8, aynı-sahne eğitiminin (C4: 96,6) çoğuna ulaşıyor) ve diğer
3 bölgede de sabit kamerayı geçmeye devam ediyor. Sonuç: **iki sahneyi birlikte
eğitmek, tek-sahne modelinin en kötü durumdaki başarısızlığını gideriyor**, ekstra
maliyeti yok gibi görünüyor.

`ds3 28001–30000` bölgesi ayrıca ilginç: sabit kamera burada çok zayıf (65,8, oracle
100,0 — kamera değiştirmenin gerçekten önemli olduğu bir bölge) ve PPO üç
yapılandırmada da neredeyse tavana ulaşıyor (99,4–99,9). Kamera değiştirmenin işe
yaradığı yerde ajan bunu gerçekten yapıyor.

## C_hist — kendi geçmişi + geçiş cezası (2026-09-22, 3 tohum)

Gözlemlenen sorun: ajan çok sık kamera değiştiriyordu (ör. A'da ~her 2,8 karede
bir) — muhtemelen dedektörün kare-kare titremesine tepki, gerçek durum
değişikliğine değil. İki ekleme yapıldı: **kendi geçmişi** (gözleme "bu kamerayı
az önce seçtim mi" + "kaç adımdır bu kamerada kaldım" eklenir) ve **geçiş
cezası** (kamera değiştirince ödülden 0,1 düşülür, sadece eğitimi etkiler).
`C_hist` = C ile birebir aynı karışık eğitim (ds3+ds4) + bu iki özellik.

| Bölge | Sabit | C (önce) | **C_hist** | Oracle | Geçiş sayısı: C → C_hist |
|---|---|---|---|---|---|
| ds3 12001–14000 | 83,7 | 86,7 | 86,3 ± 1,5 | 94,7 | ~438 → ~12 |
| ds3 28001–30000 | 65,8 | 99,5 | 99,4 ± 0,3 | 100,0 | ~628 → ~5 |
| ds4 5001–10000 | 76,5 | 85,4 | 85,4 ± 0,4 | 95,1 | ~973 → ~35 |
| ds4 16001–18000 | 90,2 | 94,8 | **98,3 ± 0,3** | 100,0 | ~518 → ~11 |

**Sonuç: ✅ net kazanım.** Geçiş sayısı 4 bölgede de 28-118 kat azaldı (artık
~150-380 karede bir değişiyor, önceden 3-5 karede bir). `visible_rate` 3
bölgede tohum sapması içinde aynı, 1 bölgede belirgin iyileşme (94,8 → 98,3,
oracle farkının %67'si kapandı). Hiçbir held-out bölge kötüleşmedi — bedelsiz
bir kazanım. Not: eğitim-içi bir bölgede (ds3 30201–33875, held-out değil) hem
C hem C_hist sabit kameranın altında kalıyor (78,3 → 73,9) — önceden beri var,
bu değişiklikle ilgisi yok, ayrı bir konu.

### C_hist görselleri (seed 0)

**Yörünge + kamera seçimi** — solda cam0 arka planında gerçek 2D piksel yörüngesi
(zamanla renklendirilmiş), sağda hangi kameraların o an drone'u gördüğü (gri) ile
ajanın hangisini seçtiği (turuncu) aynı zaman ekseninde. Turuncu şeritlerin uzun ve
kesintisiz olması, geçiş sayısının 28-118× azaldığını gözle doğruluyor.

![Yörünge ve kamera seçimi](docs/figures/dataset3_dataset4_yorunge_ve_kamera_secimi.png)

**Ground-truth'a göre doğruluk** — 500 karelik kayan pencerede görünür oranı: PPO
(turuncu) vs oracle/üst sınır (yeşil dolgu) vs sabit cam0 (lacivert), sadece gerçek
etiketten. Gri bantlar eğitimde hiç görülmemiş test bölgeleri. dataset3'ün sonundaki
(~30000-34000) zayıf bölge (bkz. TRAIN_EXTRA2 notu yukarıda) burada gözle görülüyor.

![Ground-truth doğruluk karşılaştırması](docs/figures/ground_truth_dogruluk_karsilastirmasi.png)

## D3 — düzeltilmiş detektör (2026-10-05, 3 tohum)

Sabit yanlış kaynak bastırma (fcd7c7b) eklendikten sonra, D3 ile aynı bölünme ve
`latent+aux` ile yeniden önbellek üretilip koşuldu (yeni makine, 28 çekirdek).

| Held-out bölge | Eski D3 | Yeni D3 (düzeltilmiş) | Sabit | Oracle | Geçiş (eski → yeni) |
|---|---|---|---|---|---|
| ds4 5001–10000 | 85,1 ± 0,2 | 85,0 ± 0,1 | 76,5 | 95,1 | eski değer kaydedilmedi → ~850–955 |
| ds4 16001–18000 | 96,6 ± 0,7 | 96,0 ± 0,3 | 90,2 | 100,0 | ~557–792 → ~425–534 |

**Sonuç: doğruluk değişmedi.** Fark tohum sapması içinde ya da hafifçe negatif.
Cam6'nın kesinliği 0,37'den 0,74'e çıktı, ama ajanın doğruluğu bundan etkilenmedi.
Geçiş sayısı 16001–18000'de ~%25-35 azaldı, bu tek başına anlamlı bir kazanç değil.
Bu düzeltme temizlik olarak kalıyor. Ajanın asıl sınırı başka bir yerde: cam0'ın
duyarlılığı (0,38), yani drone'un ~%62'si cam0'da kaçırılıyor.

## D3 — düşük eşik (tüm kameralar, ns3/mr8) (2026-10-06, 3 tohum)

Eşikler tüm kameralarda düşürüldü (min_response 14→8, noise_sigmas 6→3). Cam0 duyarlılığı
0,38'den 0,73'e çıktı, ama diğer kameralarda sahte alarm arttı (cam1 kesinlik 0,35, cam6 0,46).
Ajan doğruluğu değişmedi, hafifçe düştü:

| Held-out bölge | D3 (önceki detektör) | D3 (düşük eşik) | Sabit | Oracle |
|---|---|---|---|---|
| ds4 5001–10000 | 85,0 ± 0,1 | 84,5 ± 0,8 | 76,5 | 95,1 |
| ds4 16001–18000 | 96,0 ± 0,3 | 95,6 ± 1,1 | 90,2 | 100,0 |

**Sonuç: kabul edilmedi.** Eşiği tüm kameralar için düşürmek ajanı yanıltıyor. Bir sonraki
deneme yalnızca cam0 için düşük eşik (diğer kameralar korunur).

Not: Önceki D3 raporunda "ajan en çok cam0'ı seçiyor" dendi, bu doğru değil. D3'te en çok
seçilen kamera cam5 (seed 0, %28), cam0 %18. Bu iddia yalnızca C_hist için geçerli.

## Cam0 düşük eşiğinin sahneler arası testi (2026-10-07)

D3'te (dataset4 içi, aynı sahne) cam0 için düşük eşik 2 bölgede de kazandırmıştı
(86,0 vs 85,0; 98,0 vs 96,0; bkz. yukarıdaki "Kalıcı yap" bölümü). Bunun gerçekten
genel bir iyileştirme mi yoksa dataset4'e özgü bir ince ayar mı olduğunu görmek için,
dataset4'te eğitilen modeli dataset3'ün (hiç görülmemiş sahne) iki bölgesinde test
ettik: 12001–14000 ve 28001–30000 (dataset3 kareleri bu amaçla ilk kez çıkarıldı).

| ds3 bölgesi | Varsayılan detektörle eğitilmiş | Cam0 düşük eşikle eğitilmiş | Sabit | Oracle |
|---|---|---|---|---|
| 12001–14000 | **86,5 ± 0,1** | 85,0 ± 0,7 | 83,7 | 94,7 |
| 28001–30000 | 99,4 ± 0,1 | 99,4 ± 0,3 | 65,8 | 100,0 |

![Cam0 düşük eşik, dataset4 içi (aynı sahne)](docs/figures/cam0_esik_ds4_aynisahne.png)
![Cam0 düşük eşik, dataset3'e geçiş (sahneler arası)](docs/figures/cam0_esik_ds3_capraz.png)

**Sonuç: genellemiyor.** 12001–14000'de düşük eşik aslında daha kötü (−1,5 puan,
gürültü payının dışında). 28001–30000'de fark yok, ama oracle zaten 100 ve sabit
kamera çok zayıf (65,8) olduğu için iki model de tavana yakın — orada ayırt edici
değil. Yani dataset4 içinde görülen +1,0/+2,0 puanlık kazanç dataset4'e özgü;
muhtemelen ns3/mr8 eşiği o sahnenin özel ışık/kontrast koşullarına göre iyi
çalışıyor, dataset3'ün farklı koşullarına aktarılmıyor.

**Çıkarım:** Tek sahneye göre ince ayarlanan bir dedektör eşiği, sahneler arası
genellemeyi garanti etmiyor. `--low-threshold-cams` özelliği kodda kalıyor (genel
ve test edilmiş), ama varsayılan olarak hiçbir kamerada açık değil — "dataset4
için iyi" demek "genel olarak iyi" demek değil, bunu ayrı ayrı doğrulamak gerekir.

## Fold B — karışık eğitimin ikinci bölünmeyle tekrarı (2026-10-07, 3 tohum)
Ayarlar C_hist ile aynı (latent+aux, own-history, switch_penalty 0.1). Üç koşu:
`foldB_mix` (ds3+ds4), `foldB_ds3only`, `foldB_ds4only`; sonuçlar `results/foldB_*`
ve `results/ledger.csv`. Yeni sahnedeki her bölge için PPO − en iyi sabit kamera (puan):

| Bölge | sadece diğer sahne | karışık |
|---|---|---|
| ds4 1–4800 | +2,5 (ds3) | −0,5 |
| ds4 10201–15800 | −2,2 (ds3) | −2,5 |
| ds4 18201–26000 | −4,4 (ds3) | −5,3 |
| ds3 1–11800 | −0,8 (ds4) | −1,0 |
| ds3 14200–27800 | −7,3 (ds4) | −7,5 |
| ds3 30201–33875 | −20,1 (ds4) | −18,6 |

**Sonuç: C'deki karışık-eğitim kazancı tekrarlamadı.** Küçük bir eğitim-verisi
dengesizliğiyle de karıştı; "işe yaramıyor" değil, "işe yaradığı güvenilir şekilde
gösterilemedi". Aynı sahnede ajan güvenilir (8/8 kazanç), yeni sahnede tutarsız.

## RELO hibrit dedektör — dedektör düzeyi kapı (2026-10-08; 2026-10-09 düzeltildi) — adil karşılaştırmada karışık-olumlu
Tasarım: `docs/superpowers/specs/2026-10-08-relo-hybrid-detector-design.md`.
SmallTargetDetector drone'u bulur, RELO-T256 (ICML 2026, hazır ağırlık, ince ayar yok)
takip eder. Ölçüm `scripts/relo_gate.py`: aynı aralıkta klasik önbellek dilimi vs RELO
önbelleği; **hit** = etiketli karelerin ≤10 px doğru yerde bulunan oranı (hit20: ≤20 px).
Hız: 6–7 kamera paralel, kamera başına ~20 kare/s (RTX 5060).

**1. pilot (ayar penceresi ds3 20001–21500, varsayılanlar):** ortalama hit 0,830 → 0,387.
RELO güveni (softmax maksimumu) yanlış hedefte de 0,7–1,0 → kaybı yakalamıyor. İki hata:
klasiğin sahte kaynağından başlayıp onu sonsuza dek izlemek (cam0, cam4) ve drone
kaybolunca kutunun 300–800 px'e şişmesi (cam2, cam5). RELO–klasik >50 px ayrıştığında
klasik ~%97 haklıydı.

**Düzeltme (commit bff0aa4):** klasiğin kalıcı izi RELO'dan >50 px uzaksa yeniden çapala;
kutu kenarı >60 px ise kayıp say. **2. pilot (aynı pencere):** hit 0,830 → 0,842
(cam4 0,907→1,000, cam5 0,829→0,996; cam0'daki hata klasik dedektörden miras).

**Ayarda kullanılmayan pencereler (ortalama, klasik → RELO):**

| Pencere | hit | hit20 | precision | Öne çıkan |
|---|---|---|---|---|
| ds4 5001–7000 | 0,866 → 0,870 | 0,871 → 0,874 | 0,771 → 0,628 | cam6 hit 1,00→0,92 |
| ds4 16001–18000 | 0,839 → 0,831 | 0,849 → 0,843 | 0,870 → 0,732 | cam0 0,64→0,70, cam2 0,78→0,88, **cam6 0,91→0,67** |
| ds3 12001–14000 | 0,828 → 0,791 | 0,844 → 0,801 | 0,904 → 0,875 | **cam3 1,00→0,73** |

ds4 cam0 recall yükseldi (0,36→0,40; 0,69→0,85) ama cam6 precision çöktü (0,72→0,25;
0,97→0,72). Doğru yerdeyken RELO daha hassas (medyan hata çoğu kamerada 2–4 px → 1–2 px).

**Kalan hata türü (teşhis):** RELO **sabit bir nesneye** kilitleniyor. ds4 cam6'da
(16, 382) — klasik dedektörün statik-kaynak bastırmasıyla öğrendiği anten; RELO orada
kalıyor (konum std 1,4 px), drone görünmezken 1485 karede çıktı veriyor (klasik 195).
ds3 cam3'te (749, 676), std 0,4 px. **Not:** bu teşhis ayar dışı pencerelere bakılarak
yapıldı; bir sonraki düzeltme bunlara göre yapılırsa bu pencereler artık "dokunulmamış"
sayılamaz, yeni bir doğrulama penceresi gerekir.

**Karar:** spec'teki kapı ölçütü (dedektör düzeyinde klasikten daha iyi) sağlanmadı →
RL koşuları başlatılmadı, kullanıcıya raporlandı.

**⚠ DÜZELTME (2026-10-09): yukarıdaki "sabit nesneye kilitleniyor" teşhisi YANLIŞ, kapı karşılaştırması adil değildi.**
Kare kare inceleme (`scripts/relo_lock_cause.py`): ds4 cam6'da RELO'nun (16, 382)'ye her
gidişi bir *yeniden çapalama* anı — RELO o sırada drone'u 0–1 px hatayla, güven 1,0 ile
takip ediyordu; hibritin içindeki klasik dedektör antende kalıcı iz verince kural RELO'yu
antene taşıdı. Neden klasik antende? Karşılaştırma önbelleği (`dataset4_fixed`) 1. kareden
üretilmişti, klasik dedektör anteni ~4600. karede bastırmayı öğrenmişti ("sıcak"); RELO
önbellekleri ise pencere başından sıfırdan ("soğuk") başladı. Aynı pencerede soğuk başlayan
klasik dedektör: antende 600 kare, cam6 hit 0,664 (RELO hibriti 0,668). Yani cam6 kaybı
tamamen soğuk başlangıçtan.

**Adil karşılaştırma (ikisi de soğuk başlangıç, `scripts/cold_gate.sh`), ortalama:**

| Pencere | hit klasik → RELO | hit20 | precision |
|---|---|---|---|
| ds4 5001–7000 | 0,847 → **0,870** | 0,852 → 0,874 | 0,727 → 0,628 |
| ds4 16001–18000 | 0,797 → **0,831** | 0,806 → 0,843 | 0,850 → 0,732 |
| ds3 12001–14000 | **0,817** → 0,791 | 0,832 → 0,801 | 0,903 → 0,875 |
| ds3 20001–21500 (ayar) | 0,811 → **0,842** | 0,832 → 0,895 | 0,970 → 0,936 |

Ayar dışı 3 pencerenin 2'sinde RELO daha iyi (+2,3, +3,4), 1'inde kötü (−2,6; tek kamera:
ds3 cam3 0,958→0,725, yalnızca 142 etiketli kare). cam6 artık eşit/daha iyi (0,664→0,668;
0,891→0,915). **Gerçek kalan zayıflıklar:** (1) precision her yerde düşük — RELO'nun "hedef
yok" kavramı yok, drone kadrajdan çıkınca da yüksek güvenle bir şey bildirmeye devam ediyor;
(2) klasik dedektörün yanlış başlatmalarını devralıyor (ds3 cam0'da klasik zaten 0,27–0,35);
(3) yeniden çapalama kuralı, klasik dedektör yanıldığında doğru takibi bozabiliyor.
Doğru yerdeyken RELO tutarlı biçimde daha hassas (medyan hata 2–4 px → 1–2 px).


## RELO hibrit dedektörle RL (2026-10-09, 3 tohum, 200k adım)
Zincir: `scripts/relo_chain.sh` (5060 masaüstü). Önbellekler 1. kareden: `dataset3_relo.npz`,
`dataset4_relo.npz` (RELO-T256, yeniden çapa 50 px, kutu sınırı 60 px). Ayarlar C_hist ile aynı
(own-history, switch_penalty 0.1). Karşılaştırmalar aynı bölge, aynı ayar; sadece dedektör/gözlem farklı.

**Dedektör, tüm veri seti (ikisi de 1. kareden, `results/gate_full_*.txt`):** ortalama hit
ds3 0,828 → **0,875**, ds4 0,814 → **0,886** (ds4 cam0 0,333 → 0,613; ds3 cam0 0,560 → 0,691).
Precision düşük: ds3 0,922 → 0,831, ds4 0,798 → 0,648 (RELO'nun "hedef yok" kavramı yok).

**C bölünmesi — held-out bölgeler** (`C_fixed` = güncel klasik önbellek, latent+aux;
`C_relo` = RELO önbelleği, latent+aux+relo; `C_relo_ozelliksiz` = RELO önbelleği, latent+aux):

| Bölge | Sabit | Oracle | C_fixed | C_relo | C_relo_ozelliksiz |
|---|---|---|---|---|---|
| ds3 12001–14000 | 83,7 | 94,7 | 87,2 ± 0,1 | 87,4 ± 0,3 | 85,6 ± 0,9 |
| ds3 28001–30000 | 65,8 | 100 | 98,9 ± 0,2 | 98,1 ± 0,7 | 98,1 ± 0,4 |
| ds4 5001–10000 | 76,5 | 95,1 | 84,8 ± 0,5 | 84,6 ± 0,4 | 82,7 ± 1,5 |
| ds4 16001–18000 | 90,2 | 100 | **97,7 ± 0,2** | 92,6 ± 3,8 | 95,0 ± 2,5 |

→ Aynı-sahne held-out'ta **kazanç yok** (3 bölge eşit, ds4 16001–18000'de kayıp, tohum varyansı
yüksek). Geçiş sayısı RELO'yla belirgin az (ör. 68 → 27, 17 → 9). Eğitim bölgelerinde RELO
belirgin iyi (ds3 30201–33875: 71,6 → 88,5 / 98,3; ds4 1–4800: 78,1 → 83,6) — uyum, genelleme değil.

**Fold B — held-out bölgeler** (`foldB_mix` = klasik, 2026-10-07; `foldB_relo` = RELO, latent+aux+relo).
Fold B'nin held-out'u ağırlıkla "yeni sahne" problemiydi:

| Bölge | Sabit | foldB_mix | foldB_relo | Fark |
|---|---|---|---|---|
| ds3 1–11800 | 100 | 99,0 ± 0,4 | 99,0 ± 0,2 | 0 |
| ds4 1–4800 | 73,9 | 73,4 ± 0,1 | 74,8 ± 0,9 | +1,4 |
| ds4 10201–15800 | 100 | 97,5 ± 0,8 | 99,1 ± 0,4 | +1,6 |
| ds3 14200–27800 | 99,9 | 92,4 ± 0,4 | 96,3 ± 1,0 | +3,9 |
| ds4 18201–26000 | 100 | 94,7 ± 0,2 | 96,7 ± 1,0 | +2,0 |
| ds3 30201–33875 | 86,3 | 67,7 ± 1,7 | 79,1 ± 3,1 | **+11,4** |

→ 6 held-out bölgenin 5'inde iyileşme, hiçbirinde kayıp; en kötü bölgedeki açık sabit kameraya
göre −18,6'dan −7,2 puana indi. Hâlâ 3 bölgede sabit kameranın altında.

**Yorum:** RELO dedektör düzeyinde net daha iyi; RL'de etkisi bölünmeye bağlı — fold B'de (yeni
sahneye genelleme) tutarlı kazanç, C'de (aynı sahnenin görülmemiş bölümleri) kazanç yok.
RELO özelliklerinin (güven/kutu/bayrak) katkısı belirsiz: C'de 2 bölgede +1,8/+1,9, 1 bölgede −2,4
(varyans yüksek). Tek tohum seti ve iki bölünme; fold B kazancı için ikinci bir doğrulama gerekir.

## RELO kazancının tekrarı (2026-10-10, 3 tohum, 200k adım) — kısmen tekrarlandı
Zincir: `scripts/relo_replicate.sh` (5060). Ayarlar önceki RELO RL koşularıyla aynı (own-history,
switch_penalty 0.1); önbellekler aynı (`dataset{3,4}_{fixed,relo}.npz`).

**1) Fold B, yeni tohumlar 3/4/5** (`foldB_mix_s345` klasik, latent+aux; `foldB_relo_s345` RELO,
latent+aux+relo). İlk tohum seti (0/1/2) yanında, 6 tohumun ortalaması sağda:

| Bölge | Sabit | mix 0-2 | relo 0-2 | mix 3-5 | relo 3-5 | Fark 3-5 | Fark (6 tohum) |
|---|---|---|---|---|---|---|---|
| ds3 1–11800 | 100 | 99,0 | 99,0 | 99,0 ± 0,2 | 96,2 ± 4,7 | −2,8 | −1,4 |
| ds4 1–4800 | 73,9 | 73,4 | 74,8 | 72,6 ± 1,6 | 73,4 ± 2,9 | +0,8 | +1,1 |
| ds4 10201–15800 | 100 | 97,5 | 99,1 | 97,6 ± 0,8 | 98,6 ± 1,0 | +1,0 | +1,3 |
| ds3 14200–27800 | 99,9 | 92,4 | 96,3 | 91,5 ± 1,0 | **95,1 ± 1,1** | **+3,6** | **+3,8** |
| ds4 18201–26000 | 100 | 94,7 | 96,7 | 95,2 ± 0,2 | 93,6 ± 1,6 | −1,6 | +0,2 |
| ds3 30201–33875 | 86,3 | 67,7 | 79,1 | 69,0 ± 0,3 | 69,7 ± 8,7 | +0,7 | +6,0 |

→ İlk setteki **+11,4 tekrarlanmadı** (yeni tohumlar: 79,6 / 71,3 / 58,3, varyans çok yüksek).
Tutarlı tek kazanç ds3 14200–27800 (+3,9 → +3,6). RELO koşularında bir tohum çökebiliyor:
tohum 5 eğitim bölgesinde bile kötü (12001–14000: 77,8, klasik ~87) → bu genelleme değil,
**optimizasyon kararsızlığı**.

**2) Saf yeni sahne** (eğitimde hedef sahneden hiç kare yok):

| Yön | Sabit | Eski (A2/B2, latent+aux, own-history yok) | X_fixed (klasik) | X_relo (RELO) | Tohumlar (relo) |
|---|---|---|---|---|---|
| ds3 → ds4 1–26000 | 87,8 | 87,9 | **91,2 ± 0,4** | 89,4 ± 4,6 | 92,4 / 82,9 / 92,9 |
| ds4 1–26000 → ds3 | 94,1 | 94,4 | 92,5 ± 0,6 | **97,1 ± 0,6** | 96,5 / 96,9 / 97,9 |

→ ds4→ds3'te RELO **net ve tutarlı** kazanç (+4,6, üç tohum da sabit kameranın üstünde; klasik
sabit kameranın altında kalıyor). ds3→ds4'te iki tohum klasikten biraz iyi (+1,2/+1,7), biri çöktü
(82,9) → ortalama eşit/altında. Not: own-history + geçiş cezasıyla klasik de ds3→ds4'te eski
A2'den belirgin iyi (87,9 → 91,2), yani ayar değişikliği tek başına bu yönde kazanç sağladı.

**Yorum:** RELO'nun RL'e katkısı gerçek ama ilk tahmin ettiğimizden küçük ve kararsız.
Kazanç görülen yerler: ds4→ds3 saf yeni sahne (+4,6) ve ds3 14200–27800 (iki tohum setinde de
+3,6–3,9). Ana sorun RELO koşularında tek tohum çöküşleri (3 koşuda 1'er tohum). Olası nedenler
(doğrulanmadı): RELO özelliklerinin dağılımı (güven çoğunlukla 1,0'a doymuş, kutu boyutu
sahneye göre değişiyor) ya da RELO önbelleğinde precision düşüklüğü (drone yokken de çıktı) →
gürültülü ödül sinyali. Sonraki adım: çöken tohumların (foldB_relo_s345 tohum 5,
X_ds3_to_ds4_relo tohum 1) kamera izlerini incelemek; latent+aux (özelliksiz) RELO ile karşılaştırmak.

## Çöken tohumlar ve RELO özellik ablasyonu (2026-10-10) — kazanç konumdan, özellikler zararlı
**Çöken tohumların incelemesi** (`scripts/relo_collapse.py`, izler `outputs/*/seed_*/trace_*.csv`):

1. **foldB_relo_s345 tohum 5 — hayalet kameraya yapışma.** Fold B'de ds3 (6 kamera) ds4'e (7)
   dolgulanıyor; tohum 5 dolgu kamerasını (-1) seçip uzun süre bırakmıyor: ds3 1–11800'de
   1. kareden itibaren 1128 kare, eğitim bölgesinde 358, ds3 30201–33875'te 792 kare (her biri
   tek kesintisiz bölüm). Bu sürenin %78–100'ünde başka bir kamerada drone görünür.
   Own-history + geçiş cezası "kal" eğilimini güçlendiriyor; hayalete bir kez düşen ajan çıkamıyor.
   RELO'ya özgü değil (klasik koşularda da hayalet payı %0–1,4 görülüyor) ama bir tohumu çökertebiliyor.
   `pad_cameras` belgesindeki "hayalet seçmek öğrenmeyi bozmaz" varsayımı bu yüzden tam doğru değil.
2. **X_ds3_to_ds4_relo tohum 1 — yanlış alarmlı kameralara güven.** Kaçırılan karelerin %98'inde
   seçilen kamerada RELO takipte ve klasik ateşlemiş, ama drone orada yok. ds4'te drone
   görünmezken RELO takipte oranı: cam6 %81, cam3 %69, cam1 %46, cam2 %45 (klasik ateşleme
   oranları benzer: %82/%76/%51/%57). **RELO güveni bilgi taşımıyor:** hem isabet hem kaçırılan
   karelerde medyan 1,00.

**Ablasyon:** RELO önbelleği + latent+aux (güven/kutu/bayrak yok), `scripts/relo_ablation.sh`:

| Yön | Sabit | Klasik | RELO + özellikler | RELO özelliksiz |
|---|---|---|---|---|
| ds3 → ds4 1–26000 | 87,8 | 91,2 ± 0,4 | 89,4 ± 4,6 (92,4/82,9/92,9) | 91,2 ± 0,7 (92,1/90,5/91,1) |
| ds4 1–26000 → ds3 | 94,1 | 92,5 ± 0,6 | 97,1 ± 0,6 | **97,2 ± 0,9** (98,4/96,9/96,2) |

→ ds4→ds3 kazancı (+4,7) özellikler olmadan da aynen duruyor: **kazanç RELO'nun konumundan**
(daha iyi kırpım/izler), gözleme eklenen RELO özelliklerinden değil. ds3→ds4'te özellikler
çıkarılınca çöküş kayboluyor (klasikle eşit). C bölünmesindeki `C_relo_ozelliksiz` sonucu da
özelliklerin net katkısı olmadığını gösteriyordu. **Karar önerisi:** RELO önbelleği + latent+aux
varsayılan olsun; `latent+aux+relo` bırakılsın (ya da güven yerine anlamlı bir sinyal bulunsun).
Ayrı düzeltme önerisi: kayıtta olmayan kameraları (hayalet dahil; `recording` senkron tablosundan,
nedensel) eylem olarak maskelemek.

## Bilinen sınırlamalar (henüz çözülmedi)
- "Sabit kamera" test verisinden seçiliyor → ajana karşı iyimser bir ölçüt.
- Tek sahne çifti (dataset3, dataset4), tek bölünme seti; güven aralığı yok.
- Dedektör dataset4'te zayıf (cam0 duyarlılık 0,39, cam6 kesinlik 0,37) — bkz. rapor.
- dataset1/2 (senkronizasyon tablosu yok) ve dataset5 (2D etiket yok) hâlâ kullanılamıyor.

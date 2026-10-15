---
language: [tr, en]
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: other
tags: [behavior-analysis, multimodal, emotion, measurement-grounded, turkish]
extra_gated_prompt: >-
  Bu model yalnızca araştırma ve ticari olmayan amaçlarla paylaşılır. Erişim istemekle (1) LICENSE-WEIGHTS (CC BY-NC-SA 4.0) koşullarını ve eğitim verisi
  koşullarını (README §3) kabul ettiğinizi, (2) modeli işe alım/değerlendirme, güvenlik/gözetim, sağlık teşhisi, kimlik tanıma, yalan/niyet
  tespiti ya da rızası alınmamış kişilerin izlenmesi için KULLANMAYACAĞINIZI beyan edersiniz.
extra_gated_fields:
  Ad Soyad: text
  Kurum: text
  Kullanım amacı: text
  Yalnızca araştırma/ticari olmayan amaçla kullanacağımı kabul ediyorum: checkbox
  Kullanım dışı alanlarda (README §6) kullanmayacağımı kabul ediyorum: checkbox
---

# Davranış Analizi v2.1.1 — ölçüme dayalı, temkinli yorum üreten çok kipli model

> **Bu bir yalan tespiti, kişilik testi ya da duygu "okuma" aracı DEĞİLDİR.** Model, kısa bir video/ses penceresinde *ölçülen* yüz ve ses
> göstergelerini listeler, her göstergenin yaygın anlamını **temkinli** bir dille yazar ve söylenen içerikle birlikte genel bir duygu dağılımı verir.
> Yüz/ses ile ilgili her iddia **ölçüm dilindedir** ("ölçümde kaş indirme belirgin (kanıt), z=…"), algı iddiası değildir.

**Durum:** v2.1.1 nihai (dondurulmuş yapılandırma sha256 `74b470be…`, `yapilandirma/v211_dondurulmus.json`). Test kümeleri tek kez koşuldu; sonuç sonrası ayar yok.
**Önemli:** çıtaların çoğu KALDI (aşağıda §4); bu sürüm duygu doğruluğunu v2.1'e göre iyileştirmedi — kazanım ölçüm katmanının dürüstlüğüdür.

## 1. Amaç
Pencerelerde (görüntü + ses + transkript) şunları üretmek: (i) kodla ölçülmüş işaretler (**kanıt** / **olası** katmanları), (ii) her işaretin sözlükteki
anlamı + alternatifi (kural tabanlı), (iii) söylenen içeriğin duygusu, (iv) iki cümlelik, temkinli bir Türkçe genel yorum, (v) birleşik duygu dağılımı (`genel_duygu`).

## 2. Mimari (çıkarım hattı)
| Katman | Bileşen | Kim üretir |
|---|---|---|
| Yüz ölçümü | YuNet kutu + MediaPipe Face Landmarker blendshape (klip başına 8 kare ortalaması) | **kod** |
| Ses ölçümü | librosa (rms, sessizlik oranı, perde ort./std) | **kod** |
| Katman atama | z-skoru; nedensel kişi referansı (aynı konuşmacının ÖNCEKİ klipleri; yoksa "genel" popülasyon) → **kanıt**: yüz z≥2,6 / ses \|z\|≥2,25 + yeterli referans (≥10 klip ve ≥30 sn); **olası**: kalibre eşikler | **kod** |
| Ölçüm cümlesi | `davranis_model/olcum_dili.py` — sabit biçim, LLM'e bırakılmaz | **kod** |
| Anlam / alternatif | `anlam_cozumleme.json` sözlük kuralları (uyum iddiası KAPALI) | **kod** |
| Algı (sınıf olasılıkları) | ViT (yüz) + Whisper-large-v3-turbo (ses) → köprü MLP → 8 sınıf başlığı (`agirlikli_us05`) | öğrenilmiş |
| Söylenen içerik duygusu | Qwen2.5-7B-Instruct (QLoRA) — slot olasılıkları | öğrenilmiş |
| Yorum | Qwen2.5-7B-Instruct (QLoRA) — yalnız 2 yorum cümlesi; koşul: kodun ölçümleri | öğrenilmiş |
| Birleşik duygu (`genel_duygu`) | kural (a): davranış başlığı güveni < 0,5 ise içerik dağılımı, aksi halde davranış; argmax | **kod** |
| Güven kuralı | en olası sınıf p < 0,55 → "çıkarım yapılamıyor" (`secici=False`) | **kod** |

v2.1.1'in v2.1'den farkı: kanıt/olası katmanı artık öğrenci başlığı DEĞİL, **kodun doğrudan ölçümüdür** (MediaPipe + librosa, nedensel kişi referansı);
kapalı sinyal ihlali yapısal olarak 0'dır. Metin sınıflandırıcı (xlm-roberta-base) ve birleşim val'de denendi, kazanç <0,02 olduğu için **seçilmedi**; pakette yoktur.

Denetimler (kod): yorum–ölçüm tutarlılığı, halüsinasyon, algı dili, uyum iddiası, kapalı sinyal ihlali.

## 3. Veri ve lisanslar
| Veri | Kullanım | Koşul |
|---|---|---|
| CREMA-D | eğitim/değerlendirme (oyunculu) | araştırma / ticari olmayan |
| RAVDESS | eğitim/değerlendirme (oyunculu) | **CC BY-NC-SA 4.0 → ticari olmayan** (ShareAlike'in ağırlıklara etkisi netleştirilmemiştir) |
| MELD | eğitim (train/dev), test | GPL-3.0 (depo) + dizi içeriği telifi; yalnız araştırma |
| CMU-MOSEI | eğitim (train/val; ses+metin), test | veri lisansı doğrulanamadı; yalnız araştırma varsayımı |
| MUStARD++ | **yalnız değerlendirme** | "research purposes"; depoda LICENSE yok |
| duygu7 (FER2013 / AffectNet / RAF-DB) | ViT yüz kodlayıcı ön eğitimi | araştırma / ticari olmayan koşullar |

Taban modeller: Qwen2.5-7B-Instruct (Apache-2.0), Whisper-large-v3-turbo (MIT), MediaPipe Face Landmarker (Apache-2.0), YuNet / OpenCV Zoo (MIT), BlazeFace (Apache-2.0; `davranis_model/BLAZEFACE_LICENSE.txt`), librosa (ISC).

**Lisans (kod ve ağırlıklar ayrı):** kod (`cikarim_v211.py`, `davranis_model/`, `space/`, testler) → **MIT** (`LICENSE`); ağırlıklar (`agirliklar/`) ve model çıktıları → **CC BY-NC-SA 4.0**
(`LICENSE-WEIGHTS`; ticari olmayan, RAVDESS CC BY-NC-SA ile uyumlu). **Eğitim verisi koşulları (MELD, CMU-MOSEI, CREMA-D, RAVDESS) ayrıca geçerlidir; bu bir hukuki görüş değildir.**

**Bu depoda MELD/MOSEI/MUStARD++/CREMA-D/RAVDESS klibi, transkripti, öğretmen çıktısı ya da klip düzeyinde türevi YOKTUR.** Tek veri türevi:
`yapilandirma/referans_populasyon.json` — MELD train'den türetilmiş **toplu özet istatistik** (8625 klip; 52 yüz + 4 ses özelliğinin ortalama/std'si, z-skoru referansı); **ham veri, klip ya da metin içermez**.

## 4. Değerlendirme (gerçek veri; test kümeleri eğitimden KESİN hariç; v2 / v2.1 / v2.1.1 AYNI test örnekleri)
Aralıklar %95 bootstrap. Örneklem: MELD 798, MOSEI 800, MUStARD++ 800, oyunculu 858×3 koşul = 4972 kayıt; hata 0.

| ölçüt | çıta | v2 | v2.1 | v2.1.1 | durum |
|---|---|---|---|---|---|
| MELD test weighted-F1 (`genel_duygu`) | ≥0,55 | 0,119 | 0,490 [0,452; 0,523] | 0,483 [0,446; 0,517] | KALDI |
| MELD seçici doğruluk (kapsama) | >0,489 (çoğunluk) | 0,177 (0,73) | 0,496 (0,808) | 0,490 (0,818) [0,455; 0,528] | GEÇTİ (sınırda) |
| MOSEI Acc-2 zorunlu (çoğunluk tabanı 0,701) | ≥0,75 | 0,326¹ | 0,653 [0,617; 0,688] | 0,656 [0,620; 0,690] | KALDI (tabanın altında) |
| MUStARD++ "uyumsuz" kesinliği | ≥0,70 | F1 0,015 | tanımsız (kapalı) | tanımsız (kapalı; dev AUC 0,499) | KALDI |
| Oyunculu seçici doğruluk gerilemesi (tam koşul) | ≤0,03 | 0,857 | 0,864 (kapsama 0,437) | 0,864 (kapsama 0,437) | GEÇTİ² |
| Halüsinasyon | 0 | 0 | 0 | 0 | GEÇTİ |
| **Kapalı sinyal ihlali** | 0 | — | **1386/4972 (%27,9)** | 0/4972 | GEÇTİ |
| Yorum–ölçüm tutarsız kayıt | 0 | — | — | 0/4972 | GEÇTİ |
| Uyum iddiası | 0 | 6/800 | 0 | 0 (kapalı) | GEÇTİ |
| Kanıt tam (oyunculu, tam koşul) | ≥%15 | %27,0 | %17,0 → %16,2³ | **%10,0** [7,9; 12,1] | **KALDI** |
| İşaretli (oyunculu, tam koşul) | ≥%50 | %37,2 | %28,3 → %25,8³ | %61,9 [58,6; 65,0] | GEÇTİ⁴ |
| E1 kanıt kesinliği (yargıç) | ≥0,85 | — | — | ÖLÇÜLMEDİ | — |

¹ v2 "karar verilen" değeri. ² Kapsama yarıya düştü; `genel_duygu` oyunculu doğruluğu 0,662 (v2 argmax 0,790'ın 0,128 altında). ³ Kapalı sinyaller çıkarılınca.
⁴ Ağırlıkla "olası" katmandan gelir; olası tek başına duygu kanıtı değildir. E1 kanıt kesinliği bu koşuda yeniden ölçülmedi (kod katmanı değişmedi; önceki ölçüm yargıç-1'e dayanır, insan değerlendirici kullanılmadı).

Ek sayılar: MELD kanıt≥1 oranı 0,142 / işaretli 0,677; MUStARD++ 0,193 / 0,685; MOSEI 0,003 / 0,310 (görüntü yok); oyunculu tüm koşullar 0,070 / 0,467.
Kişi referansı yeterli oran (test alt kümesi, nedensel): MELD 582/798, MUStARD++ 597/800, oyunculu 721/858, MOSEI 26/800.

Literatür bağlamı (doğrudan kıyas DEĞİL; bu verilerde eğitimli/ince ayarlı sistemler): MELD w-F1 62–73, MOSEI Acc-2 ≈85, MUStARD++ F1 72–78.
Val'de çok karşılaştırma yapıldı; seçim sonuçları iyimser olabilir.

## 5. Dürüst sınırlar
- **Uyum (söz–davranış) iddiası kapalı.** MUStARD++ dev'de uyum kuralı AUC ≈0,50 (<0,60). Model sarkazmı/uyumsuzluğu ölçemez.
- **Yalan tespiti değildir;** kişilik/ruh hali/niyet çıkarımı yapmaz. `ruh_hali` yalnız kural koşulu sağlanırsa ve "(kanıt katmanı)" ibaresiyle yazılır.
- **Oyunculu ağırlıklı veri:** yüz/ses göstergelerinin duygu bağı büyük ölçüde oyunculu (CREMA-D/RAVDESS) veriyle öğrenildi; doğal konuşmada genelleme sınırlıdır.
- **MOSEI:** yalnız ses+metin (görüntü yok); zorunlu Acc-2 (0,656) çoğunluk tabanının (0,701) **altındadır** — anlamlı üstünlük kanıtlanmadı.
- **Kanıt katmanı seyrek (%10):** göstergelerin çoğu "olası" katmandadır ve tek başına duygu kanıtı değildir. Kanıt için aynı kişinin ≥10 önceki klibi ve ≥30 sn'si gerekir;
  tek klip girdisinde (referans yok) kanıt katmanı oluşmaz.
- **Duygu doğruluğu v2.1'e göre iyileşmedi** (MELD w-F1 0,483 vs 0,490; aralıklar örtüşür). Davranış başlığı MELD'de emin-ama-yanlış olabilir (kalibrasyon v2.2 adayı).
- **Bilinen sorun (v2.1, düzeltildi): kapalı sinyal ihlali.** v2.1 öğrenci başlığı KAPALI baş/bakış sinyallerini kanıt katmanında yazdı; 4972 kaydın 1386'sında (%27,9; MELD %76, MUStARD++ %76, oyunculu %6,5)
  göründü ve v2.1 "kanıt≥1" oranlarını şişirdi (kapalılar çıkarılınca ≈%10–12). v2.1.1'de katman kodun doğrudan ölçümüdür ve denetim kuralı vardır (0/4972). **v2.1 çıktıları kullanılmamalıdır.**
- **Türkçe:** yorum/ölçüm dili Türkçedir; içerik duygusu İngilizce verilerle (MELD/MOSEI) eğitildi. Türkçe girdide içerik duygusu doğruluğu **ölçülmedi**.
- Referans popülasyonu MELD train'dir (kamera/ışık/oyunculuk koşulları dizi çekimidir); farklı koşullarda z-skorları kayabilir.
- Girdinin ilk 10 sn'si kullanılır; eğitim klipleri ~1–5 sn'dir.

## 6. Kullanım dışı alanlar
İşe alım/değerlendirme, kredi/sigorta, adli süreç, sınır/güvenlik taraması, gözetim, çocuklar/hassas gruplar hakkında karar verme, **kimlik tanıma**; yalan/niyet/suç tespiti;
rıza alınmamış kişilerin izlenmesi; sağlık (tıbbi/psikolojik) teşhisi. Çıktılar tek başına karar girdisi değildir.

## 7. v2.2 yol haritası
Metin sınıflandırıcı (val: MELD 0,577 / MOSEI 0,850 tek başına) + davranış başlığı güven kalibrasyonu; gerçek Türkçe video değerlendirmesi; jestler; çok kişili sahneler; **kısa ifadeler (≤1 sn)** için pencere-içi zaman çözünürlüğü (3 sn ortalama kısa gülümseme gibi olayları sulandırır; RAVDESS/webcam denemelerinde gözlendi).

## 8. Kullanım
```
pip install -r requirements.txt
python cikarim_v211.py --medya klip.mp4 --yalniz-olcum        # CPU, LLM yok: ölçüm cümleleri + kural tabanlı yorum
python cikarim_v211.py --medya klip.mp4 --cikti sonuc.json    # tam hat (GPU; Qwen2.5-7B nf4 + Whisper)
python -m unittest discover -s testler                        # birim testleri (CPU, ağırlık gerekmez)
```
Kütüphane: `OlcumKatmani` (CPU) ve `Analizci` (GPU), bkz. `cikarim_v211.py`. Ağırlıklar `agirliklar/` (agirliklar.pt = LoRA + başlıklar, epoch 2; bas_basliklari.pt; olcum_basligi.pt;
duygu7_final = ViT yüz kodlayıcı). Taban modeller (Qwen2.5-7B-Instruct, Whisper-large-v3-turbo) HF'den ayrıca indirilir.
Çıkarım ortamı: eğitimde transformers 5.16.1 / torch 2.11 kullanıldı. İlk 10 sn ve `yapilandirma/` dosyaları değiştirilmemelidir.

## 9. Atıflar
MELD (Poria vd. 2018), CMU-MOSEI (Zadeh vd. 2018), MUStARD++ (Ray vd. 2022), CREMA-D (Cao vd. 2014), RAVDESS (Livingstone & Russo 2018),
Qwen2.5, Whisper (Radford vd. 2022), BlazeFace (Bazarevsky vd. 2019), MediaPipe, YuNet (Wu vd.).

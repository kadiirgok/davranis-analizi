---
title: Davranis Analizi v2.1.1
emoji: 📏
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: mit
short_description: Olcume dayali, temkinli davranis gostergeleri
startup_duration_timeout: 1h
---

# Davranış Analizi v2.1.1 — Space

> ⚠️ **Yalan tespiti değildir.** Ölçüm göstergeleri tek başına duygu kanıtı değildir.
> 🔒 **Kullanıcı videosu/sesi saklanmaz:** yalnızca işlenmek için geçici kullanılır, işlem bitince silinir; eğitime katılmaz.
> 🧪 **Yalnızca araştırma/demo amaçlıdır;** işe alım, gözetim, sağlık teşhisi, kimlik tanıma için kullanılamaz.
> Kod MIT; ağırlıklar CC BY-NC-SA 4.0 (model deposu).
> Örnek klipler (`ornek/`): RAVDESS — Livingstone, S. R. & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. Lisans: CC BY-NC-SA 4.0 (ticari olmayan, atıf + aynı lisansla paylaşım).

Gradio + ZeroGPU. **Özel (private)** model deposundan (`MODEL_REPO`, Space secret `HF_TOKEN`) kod ve ağırlıkları indirir.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MODEL_REPO` | `akadirgok/davranis-v2-1-1` | kod + ağırlık deposu |
| `HF_TOKEN` (Secret) | — | özel depo okuma |
| `DAVRANIS_4BIT` | `1` | nf4 (eğitim tabanıyla birebir) |
| `DAVRANIS_GPU_SN` | `120` | `@spaces.GPU` süresi |

Bu bir yalan tespiti aracı değildir; kullanım dışı alanlar ve sınırlar için model kartına bakın.
Demoda MELD/MOSEI/MUStARD++/CREMA-D/RAVDESS klibi yoktur; girdi yalnızca kullanıcının kendi kaydıdır ve saklanmaz.

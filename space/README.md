---
title: Davranis Analizi v2.1.1
emoji: 📏
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: other
short_description: Olcume dayali, temkinli davranis gostergeleri
startup_duration_timeout: 1h
---

# Davranış Analizi v2.1.1 — Space

Gradio + ZeroGPU. **Özel (private)** model deposundan (`MODEL_REPO`, Space secret `HF_TOKEN`) kod ve ağırlıkları indirir.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MODEL_REPO` | `akadirgok/davranis-v2-1-1` | kod + ağırlık deposu |
| `HF_TOKEN` (Secret) | — | özel depo okuma |
| `DAVRANIS_4BIT` | `1` | nf4 (eğitim tabanıyla birebir) |
| `DAVRANIS_GPU_SN` | `120` | `@spaces.GPU` süresi |

Bu bir yalan tespiti aracı değildir; kullanım dışı alanlar ve sınırlar için model kartına bakın.
Demoda MELD/MOSEI/MUStARD++/CREMA-D/RAVDESS klibi yoktur; girdi yalnızca kullanıcının kendi kaydıdır ve saklanmaz.

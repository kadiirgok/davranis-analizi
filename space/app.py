"""Davranış Analizi v2.1.1 - HF Space (Gradio, ZeroGPU).

Hızlı katman (CPU): YuNet + MediaPipe + librosa -> ölçüm cümleleri + kural tabanlı anlam/alternatif.
LLM katmanı (GPU): Whisper + öğrenci (Qwen2.5-7B QLoRA) -> iki yorum cümlesi + birleşik duygu dağılımı.
Kod ve ağırlıklar ÖZEL model deposundan (MODEL_REPO, Space secret HF_TOKEN) indirilir. Kullanıcı girdisi kalıcı olarak SAKLANMAZ.
"""
import os
import sys
import threading
import time

import gradio as gr

try:
    import spaces
    GPU = spaces.GPU
except ImportError:                                           # yerel gelistirme
    def GPU(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

MODEL_REPO = os.environ.get("MODEL_REPO", "akadirgok/davranis-v2-1-1")
TOKEN = os.environ.get("HF_TOKEN")
YEREL_KOK = os.environ.get("DAVRANIS_YEREL_KOK")             # gelistirme: depo klasoru (indirme yok)
GPU_SN = int(os.environ.get("DAVRANIS_GPU_SN", "120"))
os.environ.setdefault("DAVRANIS_4BIT", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

KOD_DESEN = ["*.py", "davranis_model/*", "*.json", "yapilandirma/*", "olcum_modelleri/*"]
_kilit = threading.RLock()
_kok = None
_olcum = None
_analizci = None
C = None                                                      # cikarim_v211 modulu


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def kod_hazirla():
    """Kod + kucuk dosyalar (agirlik YOK): hizli katman icin yeterli."""
    global _kok, C
    with _kilit:
        if _kok is None:
            if YEREL_KOK:
                kok = YEREL_KOK
            else:
                from huggingface_hub import snapshot_download
                kok = snapshot_download(MODEL_REPO, token=TOKEN, allow_patterns=KOD_DESEN)
            sys.path.insert(0, kok)
            os.environ["DAVRANIS_KOK"] = kok
            import cikarim_v211 as c
            C = c
            _kok = kok
            log("kod hazir:", kok)
    return _kok


def agirliklari_indir_arka_plan():
    """Soguk baslangici kisaltmak icin agirlik + taban modeller GPU suresi harcamadan onceden indirilir."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_REPO, token=TOKEN)
        for r in (os.environ.get("DAVRANIS_LLM", "Qwen/Qwen2.5-7B-Instruct"), os.environ.get("DAVRANIS_WHISPER", "openai/whisper-large-v3-turbo")):
            snapshot_download(r)
        log("agirliklar onbellekte")
    except Exception as e:
        log("on indirme hatasi:", type(e).__name__, str(e)[:200])


def olcum_katmani():
    global _olcum
    kod_hazirla()
    with _kilit:
        if _olcum is None:
            _olcum = C.OlcumKatmani(_kok)
    return _olcum


def bicimle_olcum(r, ham, baglam, n_gecmis):
    s = ["### Ölçümler (hızlı katman)"]
    if ham["bulunan_kare"] == 0 and r["mod"] != "sadece_kulak":
        s.append("_Görüntüde yüz ölçülemedi; yalnız ses göstergeleri değerlendirildi._")
    if not r["olcum"]:
        s.append("Eşiği geçen ölçüm yok.")
    else:
        s.append("| katman | ölçüm |\n|---|---|")
        for x in r["olcum"]:
            s.append(f"| {'kanıt' if x['katman'] == 'kanit' else 'olası'} | {x['cumle']} |")
    ref = ("kişi referansı: bu oturumdaki önceki klipler" if baglam == "kisi"
           else f"referans: genel popülasyon (kanıt katmanı için aynı kişinin ≥10 önceki klibi ve ≥30 sn gerekir; oturumda {n_gecmis} klip)")
    s.append(f"\n_{ref}_")
    if r.get("yorum"):
        s.append("\n**Kural tabanlı yorum:** " + r["yorum"])
    s.append("\n> " + C.UYARI)
    return "\n".join(s)


def hizli(video, ses, gecmis):
    """CPU: ölçüm cümleleri. Dönüş: (markdown, oturum durumu, ctx)."""
    yol = video or ses
    if not yol:
        return "Bir video ya da ses kaydı verin.", gecmis, None
    ok = olcum_katmani()
    try:
        from davranis_model import dogrudan_olcum as D
        kare, y, sure = C.medya_oku(yol)
        if kare is None and y is None:
            return "Girdide görüntü ya da ses bulunamadı.", gecmis, None
        ham = ok.ham_olcum(kare, y)
        g = gecmis if gecmis is not None else D.KisiGecmisi()
        n_once = len(g.yuz)
        ol, baglam = ok.isaretler(ham, g)
        mod = C.kip(kare is not None, y is not None)
        r = C.hizli_cikti(ol, mod, sure)
        ok.gecmise_ekle(g, ham, sure)                         # nedensel: ölçümden SONRA
        return bicimle_olcum(r, ham, baglam, n_once), g, {"yol": yol, "mod": mod, "sure": sure, "ol": ol}
    except Exception as e:
        return f"Ölçüm hatası: {type(e).__name__}: {str(e)[:200]}", gecmis, None


@GPU(duration=GPU_SN)
def llm_katmani(ctx):
    """GPU: Whisper + öğrenci. ctx hizli()'dan gelir (girdi dosyası yeniden okunur ve iş bitince silinir)."""
    global _analizci
    if not ctx:
        return ""
    t0 = time.time()
    try:
        kod_hazirla()
        with _kilit:
            if _analizci is None:
                _analizci = C.Analizci(_kok, log=log).yukle()
        kare, y, sure = C.medya_oku(ctx["yol"])
        r = _analizci.analiz(kare, y, sure, ctx["ol"], ctx["mod"], yunet=olcum_katmani().yn)
    except Exception as e:
        log("llm hatasi:", type(e).__name__, str(e)[:300])
        return f"_Yorum üretilemedi ({type(e).__name__}); hızlı katman sonucu geçerlidir._"
    finally:
        try:                                                  # girdi dosyası işleme biter bitmez silinir
            if ctx.get("yol") and os.path.exists(ctx["yol"]):
                os.remove(ctx["yol"])
        except Exception:
            pass
    if r.get("hata"):
        return f"_Yorum üretilemedi: {r['hata']}_"
    gd = r["genel_duygu"]
    s = ["### Model yorumu (LLM katmanı)"]
    if r.get("transkript"):
        s.append(f"**Söylenen (otomatik transkript):** {r['transkript']}")
    karar = gd["sinif"] if gd["secici"] else "çıkarım yapılamıyor"
    s.append(f"**Genel duygu dağılımı:** en olası sınıf **{gd['sinif']}** (p={gd['p']:.2f}); karar: **{karar}** (eşik 0,55)")
    s.append("| sınıf | olasılık |\n|---|---|\n" + "\n".join(f"| {k} | {v:.2f} |" for k, v in sorted(gd["dagilim"].items(), key=lambda kv: -kv[1])))
    gy = ((r.get("json") or {}).get("olay") or {}).get("genel_yorum")
    if gy:
        s.append("**Genel yorum:** " + gy)
    s.append(f"\n_Süre: {time.time() - t0:.0f} sn · yapılandırma {r['yapilandirma_sha256']}_")
    s.append("\n> " + r["uyari"])
    return "\n".join(s)


def sifirla():
    return None, "Oturum geçmişi sıfırlandı.", ""


ACIKLAMA = """# Davranış Analizi v2.1.1 — ölçüme dayalı, temkinli yorum
**Yalan tespiti, kişilik testi ya da duygu "okuma" aracı DEĞİLDİR.** Kısa bir video/ses penceresinde *ölçülen* yüz ve ses göstergelerini listeler;
her göstergenin yaygın anlamını temkinli yazar. Uyum (söz–davranış) iddiası kapalıdır.

* **Gizlilik:** yüklediğiniz/kaydettiğiniz klip yalnızca işlenmek için geçici olarak kullanılır, işlem biter bitmez silinmeye çalışılır; kaydedilmez, eğitime katılmaz.
  Oturumdaki önceki klipler yalnızca oturum belleğinde tutulur (kişi referansı için; sayfayı kapatınca gider).
* Girdinin ilk 10 saniyesi kullanılır. Yalnızca kendi görüntünüzü/sesinizi ya da izin aldığınız kişileri işleyin.
* **Kullanım dışı:** işe alım, güvenlik/gözetim, sağlık teşhisi, kimlik tanıma, kredi/sigorta, adli süreçler.
* Araştırma/ticari olmayan amaçlıdır; sınırlar ve veri lisansları model kartında.
"""


def arayuz():
    with gr.Blocks(title="Davranış Analizi v2.1.1") as demo:
        gr.Markdown(ACIKLAMA)
        gecmis = gr.State(None)
        ctx = gr.State(None)
        with gr.Row():
            with gr.Column():
                video = gr.Video(sources=["upload", "webcam"], label="Video (yüz + ses)")
                ses = gr.Audio(sources=["upload", "microphone"], type="filepath", label="ya da yalnız ses")
                with gr.Row():
                    btn = gr.Button("Analiz et", variant="primary")
                    sf = gr.Button("Oturumu sıfırla")
            with gr.Column():
                cikti1 = gr.Markdown()
                cikti2 = gr.Markdown()
        btn.click(hizli, [video, ses, gecmis], [cikti1, gecmis, ctx], api_name="olcum").then(
            lambda: "_LLM katmanı çalışıyor (ilk istekte soğuk başlangıç birkaç dakika sürebilir)…_", None, cikti2).then(
            llm_katmani, [ctx], [cikti2], api_name="yorum")
        sf.click(sifirla, None, [gecmis, cikti1, cikti2])
    return demo


if not YEREL_KOK:
    threading.Thread(target=agirliklari_indir_arka_plan, daemon=True).start()
demo = arayuz()
demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()

"""Davranış Analizi v2.1.1 - HF Space (Gradio, ZeroGPU).

Hızlı katman (CPU): YuNet + MediaPipe + librosa -> 3 sn pencerelerde ölçüm cümleleri + kural tabanlı anlam/alternatif.
LLM katmanı (GPU): Whisper + öğrenci (Qwen2.5-7B QLoRA) -> iki yorum cümlesi + birleşik duygu dağılımı (en çok işaretli pencere).
Kod ve ağırlıklar ÖZEL model deposundan (MODEL_REPO, Space secret HF_TOKEN) indirilir. Kullanıcı girdisi kalıcı olarak SAKLANMAZ.
Cihaza duyarlı ses ölçümleri kişisel referans (>=10 konuşma penceresi, >=30 sn; oturum belleği) olmadan gösterilmez.
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


def _surumler():
    import importlib.metadata as md
    s = {}
    for p in ("torch", "torchvision", "transformers", "gradio", "spaces"):
        try:
            s[p] = md.version(p)
        except Exception:
            s[p] = None
    return s


log("surumler:", _surumler())


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


SES_DURUM_TR = {
    "okundu": "okundu",
    "sessiz": "ses akışı var ama sinyal yok (sessiz kayıt) — mikrofon/izni kontrol edin",
    "yok": "ses akışı yok",
}


def pencere_md(i, pen, s, r):
    """Tek pencere: yuz/ses durum satirlari + olcum tablosu (kural tabanli yorum ayri)."""
    t = f"{pen['t0']:.0f}–{pen['t1']:.1f} sn"
    L = [f"**Pencere {i + 1} ({t})** · yüz bulunan kare: {s['yuz_bulunan']}/8 · ses: {SES_DURUM_TR[pen['ses_durumu']]}"]
    if s["yuz_bulunan"] == 0 and pen["kare"] is not None:
        L.append("_Görüntüde yüz bulunamadı (YuNet eşiği 0,7; yüz küçük/kapalı olabilir)._")
    if not r["olcum"]:
        if s["yuz_bulunan"] > 0:
            L.append("Eşiği geçen ölçüm yok (3 sn pencere ortalaması genel referansa göre eşik altında; kısa süreli ifadeler ortalamada sulanır).")
        else:
            L.append("Eşiği geçen ölçüm yok.")
    else:
        L.append("| katman | ölçüm |")
        L.append("|---|---|")
        for x in r["olcum"]:
            L.append(f"| {'kanıt' if x['katman'] == 'kanit' else 'olası'} | {x['cumle']} |")
    return "\n".join(L)


def referans_md(o):
    n, sn = o.ses_n, o.ses_sn
    if o.ses_referansi_hazir:
        return f"_Kişisel ses referansı hazır ({n} konuşma penceresi, {sn:.0f} sn): ses ölçümleri kişiye göre._"
    return (f"⏳ **Kişisel referans oluşturuluyor:** ses ölçümleri (ses seviyesi, duraksama, perde) cihaza duyarlı olduğu için "
            f"≥{C.SES_REF_PENCERE} konuşma penceresi ve ≥{C.SES_REF_SN:.0f} sn normal konuşma toplanana kadar GÖSTERİLMEZ "
            f"({n}/{C.SES_REF_PENCERE} pencere, {sn:.0f}/{C.SES_REF_SN:.0f} sn).")


def hizli(video, ses, oturum):
    """CPU: pencereli olcum. Donus: (markdown, oturum, ctx)."""
    yol = video or ses
    if not yol:
        return "Bir video ya da ses kaydı verin.", oturum, None
    ok = olcum_katmani()
    try:
        o = oturum if oturum is not None else C.CanliOturum()
        m = C.medya_pencereleri(yol)
        if not m["pencereler"]:
            return "Girdide görüntü ya da ses bulunamadı.", o, None
        bolumler, kural, en, en_n = [], [], None, -1
        for i, pen in enumerate(m["pencereler"]):
            s = o.isle(ok, pen)
            mod = C.kip(pen["kare"] is not None, pen["ses"] is not None)
            r = C.hizli_cikti(s["ol"], mod, pen["t1"] - pen["t0"], t0=pen["t0"])
            bolumler.append(pencere_md(i, pen, s, r))
            if s["ol"]:
                kural.append(f"**Pencere {i + 1}:** {r['yorum']}")
            if len(s["ol"]) > en_n:
                en, en_n = (i + 1, pen, s, mod), len(s["ol"])
        k = m["kodek"]
        ses_b = f" ({k['ses_hz']} Hz, {k['ses_kanal']} kanal)" if k["ses_hz"] else ""
        ust = (f"### Ölçümler (hızlı katman)\nGirdi: video {k['video_kodek'] or '—'}, ses {k['ses_kodek'] or '—'}{ses_b}"
               f" · {len(m['pencereler'])} pencere (3 sn)\n\n" + referans_md(o) + "\n\n" + "\n\n".join(bolumler))
        uyari = "\n\n> " + C.UYARI
        ek = ("\n\n**Kural tabanlı yorum:**\n\n" + "\n\n".join(kural)) if kural else ""
        ctx = {"yol": yol, "ust": ust, "uyari": uyari}
        if en is not None and en[2]["ol"]:
            ctx.update(pencere_no=en[0], pen=en[1], ol=en[2]["ol"], mod=en[3], sure=en[1]["t1"] - en[1]["t0"])
        return ust + ek + uyari, o, ctx
    except Exception as e:
        return f"Ölçüm hatası: {type(e).__name__}: {str(e)[:200]}", oturum, None


@GPU(duration=GPU_SN)
def llm_katmani(ctx):
    """GPU: Whisper + öğrenci, EN ÇOK işaretli pencere için. Başarılıysa kural yorumu üst bölümden çıkarılır (tekrar yok).
    Dönüş: (ölçüm md, model md)."""
    global _analizci
    if not ctx:
        return gr.update(), ""
    t0 = time.time()
    try:
        if "pen" not in ctx:
            return gr.update(), "_Eşiği geçen ölçüm bulunan pencere yok; model yorumu üretilmedi (yukarıdaki sonuç geçerlidir)._"
        kod_hazirla()
        with _kilit:
            if _analizci is None:
                _analizci = C.Analizci(_kok, log=log).yukle()
        pen = ctx["pen"]
        r = _analizci.analiz(pen["kare"], pen["ses"], ctx["sure"], ctx["ol"], ctx["mod"], yunet=olcum_katmani().yn)
    except Exception as e:
        log("llm hatasi:", type(e).__name__, str(e)[:300])
        return gr.update(), f"_Model yorumu üretilemedi ({type(e).__name__}); kural tabanlı yorum geçerlidir._"
    finally:
        try:                                                  # girdi dosyası işleme biter bitmez silinir
            if ctx.get("yol") and os.path.exists(ctx["yol"]):
                os.remove(ctx["yol"])
        except Exception:
            pass
    if r.get("hata"):
        return gr.update(), f"_Model yorumu üretilemedi: {r['hata']}_"
    gd = r["genel_duygu"]
    s = [f"### Model yorumu (LLM katmanı) — pencere {ctx['pencere_no']}"]
    if r.get("transkript"):
        s.append(f"**Söylenen (otomatik transkript):** {r['transkript']}")
    karar = gd["sinif"] if gd["secici"] else "çıkarım yapılamıyor"
    s.append(f"**Genel duygu dağılımı:** en olası sınıf **{gd['sinif']}** (p={gd['p']:.2f}); karar: **{karar}** (eşik 0,55)")
    tablo = ["| sınıf | olasılık |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(gd["dagilim"].items(), key=lambda kv: -kv[1])]
    s.append("\n".join(tablo))
    gy = ((r.get("json") or {}).get("olay") or {}).get("genel_yorum")
    if gy:
        s.append("**Genel yorum:** " + gy)
    s.append(f"_Süre: {time.time() - t0:.0f} sn · yapılandırma {r['yapilandirma_sha256']}_")
    s.append("> " + r["uyari"])
    return ctx["ust"] + ctx["uyari"], "\n\n".join(s)         # model yorumu var: kural yorumu tekrar gösterilmez


def sifirla():
    return None, "Oturum ve kişisel referans silindi.", ""


ACIKLAMA = """# Davranış Analizi v2.1.1 — ölçüme dayalı, temkinli yorum
**Yalan tespiti, kişilik testi ya da duygu "okuma" aracı DEĞİLDİR.** Kısa pencerelerde *ölçülen* yüz ve ses göstergelerini listeler;
her göstergenin yaygın anlamını temkinli yazar. Uyum (söz–davranış) iddiası kapalıdır.

* **Önce ~30 sn normal konuşun (kişisel referans):** ses ölçümleri mikrofona/cihaza duyarlıdır; ≥10 konuşma penceresi (≥30 sn) toplanana kadar
  gösterilmez ve yorumda kullanılmaz. Referans yalnızca oturum belleğindedir; "Oturumu sıfırla" siler.
* **Gizlilik:** yüklediğiniz/kaydettiğiniz klip yalnızca işlenmek için geçici olarak kullanılır, işlem biter bitmez silinmeye çalışılır; kaydedilmez, eğitime katılmaz.
* İlk 30 saniye 3 sn'lik pencerelerle işlenir. Yalnızca kendi görüntünüzü/sesinizi ya da izin aldığınız kişileri işleyin.
* **Kullanım dışı:** işe alım, güvenlik/gözetim, sağlık teşhisi, kimlik tanıma, kredi/sigorta, adli süreçler.
* Araştırma/demo amaçlıdır; sınırlar ve veri lisansları model kartında.
"""


def arayuz():
    with gr.Blocks(title="Davranış Analizi v2.1.1") as demo:
        gr.Markdown(ACIKLAMA)
        oturum = gr.State(None)
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
        btn.click(hizli, [video, ses, oturum], [cikti1, oturum, ctx], api_name="olcum").then(
            lambda: "_Model katmanı çalışıyor (ilk istekte soğuk başlangıç birkaç dakika sürebilir)…_", None, cikti2).then(
            llm_katmani, [ctx], [cikti1, cikti2], api_name="yorum")
        sf.click(sifirla, None, [oturum, cikti1, cikti2])
    return demo


if not YEREL_KOK:
    threading.Thread(target=agirliklari_indir_arka_plan, daemon=True).start()
demo = arayuz()
demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()

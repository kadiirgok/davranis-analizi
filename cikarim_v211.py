"""Davranis Analizi v2.1.1 - CIKARIM (kutuphane + komut satiri).

Iki katman:
  OlcumKatmani  (CPU, LLM yok): YuNet kutu + MediaPipe Face Landmarker blendshape (8 kare ortalamasi) + librosa ses ozellikleri -> z -> kanit/olasi
                katmani -> olcum cumleleri + (kural tabanli) anlam/alternatif. Egitimdeki 'dogrudan olcum' ile AYNI kod (davranis_model/dogrudan_olcum.py).
  Analizci      (GPU onerilir): ViT + Whisper + koprü MLP (sinif basligi, 'agirlikli_us05') + Qwen2.5-7B (QLoRA) -> iki yorum cumlesi, icerik
                duygusu olasiliklari ve birlesik duygu dagilimi (`genel_duygu`).

genel_duygu (dondurulmus yapilandirma 74b470be): davranis basligi olasiligi P; en yuksek P < tau (0,5) ve icerik gecerliyse icerik dagilimi,
aksi halde P. argmax alinir; en yuksek olasilik < 0,55 ise `secici = False` (karar verilmis sayilmaz).

Olcum dili: yuz/ses iddialari ALGI iddiasi degil OLCUM'dur ('olcumde ... z=...'). Bu bir yalan tespiti degildir.

Komut satiri:
  python cikarim_v211.py --medya klip.mp4 [--yalniz-olcum] [--cihaz cuda] [--cikti sonuc.json]
Ortam: DAVRANIS_LLM (varsayilan Qwen/Qwen2.5-7B-Instruct), DAVRANIS_WHISPER (openai/whisper-large-v3-turbo), DAVRANIS_4BIT (1: nf4; egitim tabaniyla birebir).
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

KOK = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DAVRANIS_KOK", KOK)
if KOK not in sys.path:
    sys.path.insert(0, KOK)

SINIFLAR_8 = ["Korku", "Mutluluk", "Notr", "Ofke", "Sakin", "Saskinlik", "Tiksinme", "Uzuntu"]
ICERIK_SINIF = {"korku": "Korku", "mutluluk": "Mutluluk", "notr": "Notr", "ofke": "Ofke", "sakin": "Sakin", "saskinlik": "Saskinlik", "tiksinme": "Tiksinme", "uzuntu": "Uzuntu"}
CIKARIM_YOK = "çıkarım yapılamıyor"
ICERIK_ADAY = ("notr", "saskinlik", "sakin", "mutluluk", "ofke", "yok", "uzuntu", "tiksinme", "korku")
MAKS_SN = 10.0                 # girdinin ilk 10 sn'si kullanilir (egitim: ~3-5 sn'lik klipler)
KARE_N = 8
UYARI = "Yalan tespiti değildir; ölçüm göstergeleri tek başına duygu kanıtı değildir."


def yol(*p):
    return os.path.join(KOK, *p)


def yapilandirma():
    """Dondurulmus yapilandirma (sha256 on eki 74b470be...). Dosya degistirilmemelidir."""
    import hashlib
    with open(yol("yapilandirma", "v211_dondurulmus.json"), "rb") as f:
        b = f.read()
    cfg = json.loads(b)
    cfg["_sha256"] = hashlib.sha256(b).hexdigest()
    return cfg


# ---------------------------------------------------------------------------------------------- girdi okuma
def medya_oku(dosya, maks_sn=MAKS_SN):
    """Video/ses dosyasi -> (kareler [K,H,W,3] uint8 RGB | None, ses float32 16 kHz | None, sure_sn). K <= KARE_N esit aralikli kare; ilk maks_sn."""
    import av
    kareler, ses = [], []
    with av.open(dosya) as c:
        toplam = float(c.duration) / 1e6 if c.duration else None
        if c.streams.video:
            for fr in c.decode(video=0):
                if fr.time is not None and fr.time > maks_sn:
                    break
                kareler.append(fr.to_ndarray(format="rgb24"))
    with av.open(dosya) as c:
        if c.streams.audio:
            rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
            for fr in c.decode(audio=0):
                if fr.time is not None and fr.time > maks_sn:
                    break
                ses += [x.to_ndarray().reshape(-1) for x in rs.resample(fr)]
    y = np.concatenate(ses).astype(np.float32)[: int(maks_sn * 16000)] if ses else None
    if kareler:
        idx = np.linspace(0, len(kareler) - 1, KARE_N).round().astype(int)
        kare = np.stack([kareler[i] for i in idx])
    else:
        kare = None
    sure = min(maks_sn, toplam) if toplam else (len(y) / 16000.0 if y is not None else 0.0)
    if y is not None and sure <= 0:
        sure = len(y) / 16000.0
    return kare, y, float(sure)


# ---------------------------------------------------------------------------------------------- hizli katman (CPU)
class OlcumKatmani:
    """LLM'siz katman. Egitim/test olcumuyle ayni islem: YuNet(0,7/0,3/50) kutu -> %15 pay -> 224 + 56 px replicate -> MediaPipe blendshape."""

    def __init__(self, kok=KOK):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.cv2, self.mp = cv2, mp
        self.yn = cv2.FaceDetectorYN.create(os.path.join(kok, "olcum_modelleri", "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.7, 0.3, 50)
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=os.path.join(kok, "olcum_modelleri", "face_landmarker.task")), running_mode=vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, output_face_blendshapes=True))
        from davranis_model import dogrudan_olcum as D
        self.D = D
        with open(os.path.join(kok, "yapilandirma", "referans_populasyon.json"), encoding="utf-8") as f:
            self.pop = json.load(f)

    def _kirp(self, rgb):
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        self.yn.setInputSize((w, h))
        _, f = self.yn.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
        if f is None or not len(f):
            return None
        j = int(np.argmax(f[:, 2] * f[:, 3] * f[:, 14]))
        x, y, bw, bh = f[j, :4]
        x0, y0, x1, y1 = max(0, x - .15 * bw), max(0, y - .15 * bh), min(w, x + bw + .15 * bw), min(h, y + bh + .15 * bh)
        c = rgb[int(y0):int(y1), int(x0):int(x1)]
        if c.size == 0:
            return None
        return cv2.copyMakeBorder(cv2.resize(c, (224, 224), interpolation=cv2.INTER_CUBIC), 56, 56, 56, 56, cv2.BORDER_REPLICATE)

    def ham_olcum(self, kareler, ses):
        """kareler [K,H,W,3] | None, ses 1-D | None -> {'yuz': 52 | None, 'ses': 4, 'bulunan_kare': int}."""
        sk = []
        for fr in (kareler if kareler is not None else []):
            f = self._kirp(fr)
            if f is None:
                continue
            r = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
            if r.face_blendshapes:
                sk.append([q.score for q in r.face_blendshapes[0]])
        yuz = np.mean(sk, 0) if sk else None
        return {"yuz": yuz, "ses": np.asarray(self.D.ses_ozellik(ses) if ses is not None else [np.nan] * 4, float), "bulunan_kare": len(sk)}

    def isaretler(self, ham, gecmis=None):
        """-> (isaretler[], baglam). gecmis: dogrudan_olcum.KisiGecmisi (ayni kisinin ONCEKI klipleri) ya da None ('genel' baglam; kanit katmani yok).
        NOT: cagridan SONRA gecmis.ekle(...) yapilmalidir (nedensel)."""
        ol = self.D.isaretler(ham["yuz"], ham["ses"], self.pop, gecmis)
        return ol, (ol[0]["baglam"] if ol else ("kisi" if (gecmis is not None and gecmis.yeterli()) else "genel"))

    def gecmise_ekle(self, gecmis, ham, sure_sn):
        gecmis.ekle(ham["yuz"], ham["ses"], sure_sn)


def kip(kareler_var, ses_var):
    return "tam" if kareler_var and ses_var else "sadece_goz" if kareler_var else "sadece_kulak"


def katman_listeleri(ol, mod, SB, HD):
    """dogrudan_katman.listeler ile AYNI: isaretler -> (kanit [(ifade,1.0)], olasi [(ifade,1.0)], yokluk)."""
    from davranis_model import dogrudan_katman as DK
    return DK.listeler(ol, mod, SB, HD.AK.yokluk_isaretleri())


def hizli_cikti(ol, mod, sure_sn, HD=None, SB=None, imza=None):
    """LLM'siz sonuc: olcum cumleleri + kural tabanli parca yorumu (anlam/alternatif). Uyum iddiasi KAPALI."""
    if HD is None:
        import hedef_v2 as HD
    if SB is None:
        SB, imza = HD.sablon_yukle()
    K, O, Y = katman_listeleri(ol, mod, SB, HD)
    t = f"0.0-{sure_sn:.1f}"
    sonuc = {"t": t, "mod": mod, "olcum": [{"sinyal": x["sinyal"], "z": x["z"], "yon": x["yon"], "katman": x["katman"], "baglam": x["baglam"], "cumle": x["cumle"]} for x in ol],
             "kanit": [i for i, _ in K], "olasi": [i for i, _ in O], "yorum": None, "uyari": UYARI}
    olas = {s: 1.0 / len(SINIFLAR_8) for s in SINIFLAR_8}
    p = {"klip_id": "cikarim_0", "t": t, "etiket": "Notr", "olas": olas, "mod": mod, "transkript": "", "yokluk": [], "soz_duygu": None, "icerik_duygusu": "yok"}
    try:
        zc, olay, _ = HD.olay_ham(SB, imza, [p], olcumler=[K], olasi=[O], guven_kurali=False)
        sonuc["yorum"] = zc[0]["yorum"]
        sonuc["davranis"] = zc[0]["davranis"]
        sonuc["genel_yorum_kural"] = olay.get("genel_yorum")
    except Exception as e:                                           # kural katmani hatasi olcum cikisini dusurmesin
        sonuc["yorum_hatasi"] = f"{type(e).__name__}: {str(e)[:160]}"
    return sonuc


# ---------------------------------------------------------------------------------------------- birlesik duygu
def soft_icerik(slot_p):
    """slot_p: parca basina {aday: logp|None}. -> (ic[8], icw=1-P(yok), gecerli). (j10_val_teshis.soft_ic ile AYNI)"""
    ic, w, n = np.zeros(8), 0.0, 0
    for sp in slot_p or []:
        if not isinstance(sp, dict) or "hata" in sp:
            continue
        pr = {c: (float(np.exp(v)) if v is not None else 0.0) for c, v in sp.items()}
        tot = sum(pr.values())
        if tot <= 0:
            continue
        pr = {c: v / tot for c, v in pr.items()}
        gv = 1.0 - pr.get("yok", 0.0)
        if gv > 1e-6:
            for c, v in pr.items():
                if c in ICERIK_SINIF:
                    ic[SINIFLAR_8.index(ICERIK_SINIF[c])] += v / gv
        w += gv
        n += 1
    if n == 0:
        return np.zeros(8), 0.0, False
    return ic / n, w / n, True


def genel_duygu(p_baslik, slot_p, tau=0.5, esik=0.55):
    """Dondurulmus kural (a): davranis guveni < tau ve icerik gecerliyse icerik dagilimi; aksi halde davranis. -> dict."""
    P = np.asarray(p_baslik, float)
    ic, icw, gec = soft_icerik(slot_p)
    var = bool(gec and icw > 0)
    kaynak = "icerik" if (P.max() < tau and var) else "davranis"
    Q = ic if kaynak == "icerik" else P
    i = int(np.argmax(Q))
    return {"sinif": SINIFLAR_8[i], "p": float(Q[i]), "secici": bool(Q[i] >= esik), "kaynak": kaynak, "icerik_var": var,
            "dagilim": {s: round(float(q), 4) for s, q in zip(SINIFLAR_8, Q)}, "davranis_basligi": {s: round(float(q), 4) for s, q in zip(SINIFLAR_8, P)}}


# ---------------------------------------------------------------------------------------------- LLM yamasi (icerik slot olasiliklari)
def _yakala(self, logits, pkv, ek_slot):
    """icerik_duygusu slotunda aday degerlerin (tam dizi) log-olasiligi; onbellek KOPYALANIR (uretim akisi etkilenmez)."""
    import copy
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    ids_ = {c: self._ids(c + '"') for c in ICERIK_ADAY}
    ilk = {c: float(lp[i[0]].exp()) for c, i in ids_.items()}
    sk = {}
    for c, i in ids_.items():
        if ilk[c] < 1e-3:
            sk[c] = None
            continue
        if len(i) == 1:
            sk[c] = float(lp[i[0]])
            continue
        kopya = copy.deepcopy(pkv)
        o = self.llm(input_ids=torch.tensor([i[:-1]], device=logits.device), past_key_values=kopya, use_cache=True)
        l2 = torch.log_softmax(o.logits[0].float(), -1)
        sk[c] = float(lp[i[0]]) + sum(float(l2[t - 1, i[t]]) for t in range(1, len(i)))
    ek_slot.append(sk)


def _yakala_guvenli(self, logits, pkv, ek_slot):
    try:
        _yakala(self, logits, pkv, ek_slot)
    except BaseException as e_:                                       # yakalama uretimi ASLA bozmaz
        ek_slot.append({"hata": f"{type(e_).__name__}: {str(e_)[:120]}"})


def yama_uygula(M):
    """DavranisModel.uret_sablon: slot dongusunden once icerik olasiliklarini yakalar; ek['slot_p'] dondurur (test kernel'indekiyle AYNI yama)."""
    import inspect
    import textwrap
    src = inspect.getsource(M.DavranisModel.uret_sablon)
    a = "            else:\n                for _ in range(slot_max):"
    assert a in src, "uret_sablon slot dongusu bulunamadi"
    src = src.replace(a, "            else:\n                _yakala_guvenli(self, logits, pkv, ek_slot)\n                for _ in range(slot_max):", 1)
    b = "        E, _, algi, bilgi = self.girdi_kur(ornek)"
    assert b in src
    src = src.replace(b, "        ek_slot = []\n" + b, 1)
    c = '"icerik": list(ut.degerler),'
    assert c in src
    src = src.replace(c, '"icerik": list(ut.degerler), "slot_p": ek_slot,', 1)
    M.__dict__["_yakala"] = _yakala
    M.__dict__["_yakala_guvenli"] = _yakala_guvenli
    M.__dict__["ICERIK_ADAY"] = ICERIK_ADAY
    exec(compile(textwrap.dedent(src), "uret_sablon_yama", "exec"), M.__dict__)
    M.DavranisModel.uret_sablon = M.__dict__["uret_sablon"]


def yunet_kirp(kare, yn, S=256, PAY=1.6):
    """kare uint8 torch [K,3,H,W] RGB -> (kirpinti uint8 torch [K,3,S,S], bulunan). En buyuk x guvenli yuz; bulunamayan kare kare-pad+resize. (ozellik cikarimiyla AYNI)"""
    import cv2
    import torch
    a = kare.cpu().numpy().astype(np.uint8)
    K, _, H, W = a.shape
    yn.setInputSize((W, H))
    cikti, n = np.zeros((K, 3, S, S), np.uint8), 0
    for i in range(K):
        bgr = np.ascontiguousarray(a[i].transpose(1, 2, 0)[:, :, ::-1])
        _, f = yn.detect(bgr)
        c = None
        if f is not None and len(f):
            j = int(np.argmax(f[:, 2] * f[:, 3] * f[:, 14]))
            x, y, w, h = f[j, :4]
            cx, cy, sz = x + w / 2, y + h / 2, max(w, h) * PAY
            pad = int(sz) + 2
            b2 = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0 = int(round(cx - sz / 2)) + pad, int(round(cy - sz / 2)) + pad
            c = b2[y0:y0 + int(sz), x0:x0 + int(sz)]
            c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA) if c.size else None
        if c is not None:
            n += 1
        else:
            m_ = max(H, W)
            tuval = np.zeros((m_, m_, 3), np.uint8)
            tuval[(m_ - H) // 2:(m_ - H) // 2 + H, (m_ - W) // 2:(m_ - W) // 2 + W] = bgr
            c = cv2.resize(tuval, (S, S), interpolation=cv2.INTER_AREA)
        cikti[i] = c[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(cikti), n



# ---------------------------------------------------------------------------------------------- LLM katmani
class Analizci:
    """Tam hat. yukle() agir (7B + Whisper + ViT). Kullanim: a = Analizci(kok).yukle(); sonuc = a.analiz('klip.mp4')."""

    def __init__(self, kok=KOK, agirlik_dizini=None, cihaz=None, nf4=None, llm=None, whisper=None, log=print, egitilmis=True):
        self.kok = kok
        self.ag = agirlik_dizini or os.path.join(kok, "agirliklar")
        self.llm_adi = llm or os.environ.get("DAVRANIS_LLM", "Qwen/Qwen2.5-7B-Instruct")
        self.whisper_adi = whisper or os.environ.get("DAVRANIS_WHISPER", "openai/whisper-large-v3-turbo")
        self.nf4 = (os.environ.get("DAVRANIS_4BIT", "1") == "1") if nf4 is None else nf4
        self.cihaz = cihaz
        self.log = log
        self.cfg = yapilandirma()
        self.egitilmis = egitilmis             # False: YALNIZ akis testi (kucuk LLM, rastgele baslik; sonuclar ANLAMSIZ)
        self.m = None

    def yukle(self):
        import torch
        from transformers import AutoTokenizer
        t0 = time.time()
        cuda = torch.cuda.is_available()
        self.cihaz = self.cihaz or ("cuda" if cuda else "cpu")
        gpu = self.cihaz.startswith("cuda")
        if self.nf4 and not gpu:
            self.log("[uyari] nf4 icin GPU gerekir; fp32/bf16 CPU'ya dusuluyor")
            self.nf4 = False
        M = importlib.import_module("davranis_model.modeling_davranis")
        import hedef_v2 as HD
        self.M, self.HD = M, HD
        kw = {"llm_nf4": True} if self.nf4 else ({"llm_dtype": torch.float16} if gpu else {"llm_dtype": torch.float32})
        self.log(f"[yukle] llm={self.llm_adi} whisper={self.whisper_adi} nf4={self.nf4} cihaz={self.cihaz}")
        m = M.DavranisModel.parcalardan(os.path.join(self.ag, "duygu7_final"), self.whisper_adi, self.llm_adi, blaze_dizini=os.path.join(self.kok, "davranis_model"),
                                        lora_r=16, lora_alpha=32, lora_dropout=0.05, **kw)
        m.tokenizer_ayarla(AutoTokenizer.from_pretrained(self.llm_adi), AutoTokenizer.from_pretrained(self.whisper_adi))
        ob = torch.load(os.path.join(self.ag, "olcum_basligi.pt"), map_location="cpu", weights_only=False)
        m.olcum_basligi.load_state_dict(ob["state"], strict=False)
        m.populasyon_normu_ayarla(ob["pop_goz"], ob["pop_kulak"], ob.get("pop_bolge"))
        m.config.kanit_p_min, m.config.olasi_p_min = ob.get("kanit_p_min", m.config.kanit_p_min), ob.get("olasi_p_min", m.config.olasi_p_min)
        if ob.get("bas_norm") is not None:
            m.config.bas_norm = ob["bas_norm"]
        ks = os.path.join(self.ag, "kanit_sozlugu.json")
        if os.path.exists(ks):
            m.config.kanit_sozlugu = json.load(open(ks, encoding="utf-8")).get("izinli")
        if self.egitilmis:
            ck = torch.load(os.path.join(self.ag, "agirliklar.pt"), map_location="cpu", weights_only=False)
            sd = dict(m.named_parameters())
            eksik = [n for n in ck["state"] if n not in sd]
            assert not eksik, f"agirlikta modelde olmayan tensor: {eksik[:3]}"
            for n, v in ck["state"].items():
                sd[n].data.copy_(v.to(sd[n].device, sd[n].dtype))
            if "pop" in ck:
                m.populasyon_normu_ayarla(ck["pop"]["goz"], ck["pop"]["kulak"], ck["pop"].get("bolge"))
                m.config.kanit_p_min, m.config.olasi_p_min = ck.get("kanit_p_min", m.config.kanit_p_min), ck.get("olasi_p_min", m.config.olasi_p_min)
                if ck.get("kanit_sozlugu") is not None:
                    m.config.kanit_sozlugu = ck["kanit_sozlugu"]
                if ck.get("bas_norm") is not None:
                    m.config.bas_norm = ck["bas_norm"]
            self.log(f"[yukle] egitilmis agirlik: {len(ck['state'])} tensor; adim {ck['durum'].get('adim')} epoch {ck['durum'].get('epoch')}")
            # kopru (algi havuzu) + secilen sinif basligi (dondurulmus yapilandirma: baslik)
            self.havuz, sb0, H = self._kopru(ck)
            bh = torch.load(os.path.join(self.ag, "bas_basliklari.pt"), map_location="cpu", weights_only=False)
            self.sb = torch.nn.Linear(H, len(m.config.siniflar))
            self.sb.load_state_dict(bh[self.cfg["baslik"]])
            del ck
        else:
            self.log("[UYARI] egitilmemis mod: yalniz akis/gecikme testi; cevap kalitesi ANLAMSIZ")
            H = m.llm.config.hidden_size
            self.sb = torch.nn.Linear(H, len(m.config.siniflar))
            self.havuz = lambda o: m.algi(o["goz"], o["kulak"], o["goz_var"], o["ses_var"]).float().mean(1)
        yama_uygula(M)
        m.eval()
        m.llm.config.use_cache = True
        self.m = m
        self.siniflar = list(m.config.siniflar)
        self.SB, self.imza = HD.sablon_yukle()
        self._cuda_tasi()
        self.log(f"[yukle] hazir ({time.time() - t0:.0f} sn)")
        return self

    @staticmethod
    def _kopru(ck):
        """agirliklar.pt'den algi() havuzu (goz/kulak MLP + modalite/poz/yok) + mevcut sinif basligi. (test kernel'indeki kopru_kur_ ile AYNI)"""
        import torch
        st = ck["state"]
        dv, H = st["goz_mlp.0.weight"].shape[1], st["goz_mlp.0.weight"].shape[0]
        dw = st["kulak_mlp.0.weight"].shape[1]
        goz_mlp = torch.nn.Sequential(torch.nn.Linear(dv, H), torch.nn.GELU(), torch.nn.Linear(H, H))
        kulak_mlp = torch.nn.Sequential(torch.nn.Linear(dw, H), torch.nn.GELU(), torch.nn.Linear(H, H))
        goz_mlp.load_state_dict({k.split(".", 1)[1]: v.float() for k, v in st.items() if k.startswith("goz_mlp.")})
        kulak_mlp.load_state_dict({k.split(".", 1)[1]: v.float() for k, v in st.items() if k.startswith("kulak_mlp.")})
        P = {k: st[k].float() for k in ("modalite", "poz_goz", "poz_kulak", "goz_yok", "kulak_yok")}
        pop_g, pop_k = ck["pop"]["goz"].float(), ck["pop"]["kulak"].float()
        sb = torch.nn.Linear(H, st["sinif_basligi.weight"].shape[0])
        sb.load_state_dict({"weight": st["sinif_basligi.weight"].float(), "bias": st["sinif_basligi.bias"].float()})

        @torch.no_grad()
        def havuz(o):
            goz, kulak = o["goz"].float(), o["kulak"].float()
            g = goz_mlp(goz - pop_g.view(1, 1, -1)) + P["modalite"][0] + P["poz_goz"]
            k = kulak_mlp(kulak - pop_k.view(1, 1, -1)) + P["modalite"][1] + P["poz_kulak"]
            gv, sv = o["goz_var"].view(-1, 1, 1), o["ses_var"].view(-1, 1, 1)
            g = torch.where(gv, g, P["goz_yok"].expand_as(g))
            k = torch.where(sv, k, P["kulak_yok"].expand_as(k))
            return torch.cat([g, k], 1).mean(1)
        return havuz, sb, H

    def _cuda_tasi(self):
        import torch
        m, c = self.m, self.cihaz
        if not c.startswith("cuda"):
            return
        if not self.nf4:
            m.to(c)
        else:                                                         # nf4 LLM zaten cuda'da; geri kalan parametre/buffer tasinir
            for ad, mod in m.named_modules():
                if ad.startswith("llm"):
                    continue
                for pn, p in mod.named_parameters(recurse=False):
                    p.data = p.data.to(c)
                for bn, b in list(mod.named_buffers(recurse=False)):
                    mod._buffers[bn] = b.to(c)
            for _, p_ in m.llm.named_parameters():
                if p_.device.type == "cpu":
                    p_.data = p_.data.to(c)
        m.vit.half()
        m.whisper.half()

    # ---------------------------------------------------------------- analiz
    def analiz(self, kareler, ses, sure_sn, olcum_ol, mod=None, dil=None, yunet=None):
        """kareler [K,H,W,3] uint8 RGB | None; ses 1-D float 16k | None; olcum_ol: OlcumKatmani.isaretler(...) ciktisi (kod olcumu).
        -> {json, genel_duygu, transkript, ...}. Model KENDI olcumunu uretmez; kanit/olasi listesi KODUN dogrudan olcumudur."""
        import torch
        m, HD = self.m, self.HD
        kare_t = None if kareler is None else torch.from_numpy(np.ascontiguousarray(kareler)).permute(0, 3, 1, 2).float()
        ses_t = None if ses is None or len(ses) == 0 else torch.from_numpy(np.asarray(ses, np.float32))
        T = time.time()
        with torch.no_grad():
            on = m.on_hesapla([kare_t], [ses_t])
            if yunet is not None and kare_t is not None and int(on["yuz_bulunan"][0]) < kare_t.shape[0]:   # BlazeFace kacirdiysa YuNet kirpmasiyla 2. gecis (ozellik cikarimiyla AYNI)
                k2, _ = yunet_kirp(kare_t, yunet)
                on2 = m.on_hesapla([k2], [ses_t])
                if int(on2["yuz_bulunan"][0]) > int(on["yuz_bulunan"][0]):
                    on = on2
            tr = m.transkript_uret([ses_t], dil)
        g, s = bool(on["goz_var"][0]), bool(on["ses_var"][0])
        if not (g or s):
            return {"hata": "girdide ne görüntü ne ses var"}
        mod = mod or kip(g, s)
        K, O, _ = katman_listeleri(olcum_ol, mod, self.SB, HD)
        ornek = dict(goz=on["goz"], kulak=on["kulak"], goz_var=on["goz_var"], ses_var=on["ses_var"], t=[f"0.0-{float(sure_sn):.1f}"],
                     transkript=[tr[0] if mod != "sadece_goz" else ""], bolge=on["bolge"], geo=on["geo"], yuz_bulunan=on["yuz_bulunan"], kutu_olcek=on["kutu_olcek"],
                     olcum_override=[[tuple(x) for x in K]], olasi_override=[[tuple(x) for x in O]])
        dev = next(m.llm.parameters()).device
        with torch.no_grad():
            hv = self.havuz(ornek)[0]
            pb = torch.softmax(self.sb(hv), -1).tolist()
        fab = lambda probs, bilgi: HD.SablonUretici.olustur(self.SB, self.imza, ornek["t"], [mod], ornek["transkript"], probs, self.siniflar, bilgi["olcum"], bilgi["olasi"])
        amp = torch.autocast("cuda", dtype=torch.float16) if str(dev).startswith("cuda") else torch.autocast("cpu", enabled=False)
        try:
            with amp:
                metin, j, ek = m.uret_sablon(ornek, fab)
        except Exception as e:
            return {"hata": f"{type(e).__name__}: {str(e)[:200]}", "transkript": tr[0]}
        gd = genel_duygu(pb, ek.get("slot_p"), self.cfg["parametre"]["tau_tum"], self.cfg["genel_duygu_esigi"])
        return {"mod": mod, "t": ornek["t"][0], "transkript": ornek["transkript"][0], "json": j, "genel_duygu": gd, "icerik": ek.get("icerik"),
                "sinif_p_model": np.asarray(ek["sinif_p"]).reshape(-1, len(self.siniflar)).tolist(), "siniflar": self.siniflar,
                "yapilandirma_sha256": self.cfg["_sha256"][:12], "sure_sn": round(time.time() - T, 2), "uyari": UYARI}


# ---------------------------------------------------------------------------------------------- komut satiri
def main(argv=None):
    ap = argparse.ArgumentParser(description="Davranış Analizi v2.1.1 çıkarım")
    ap.add_argument("--medya", required=True, help="video (mp4/webm/...) ya da ses (wav/mp3) dosyası")
    ap.add_argument("--yalniz-olcum", action="store_true", help="yalnız hızlı katman (CPU, LLM yok)")
    ap.add_argument("--cihaz", default=None)
    ap.add_argument("--cikti", default=None, help="JSON çıktı dosyası")
    a = ap.parse_args(argv)
    kare, ses, sure = medya_oku(a.medya)
    ok = OlcumKatmani()
    ham = ok.ham_olcum(kare, ses)
    ol, _ = ok.isaretler(ham, None)
    mod = kip(kare is not None, ses is not None)
    sonuc = {"olcum_katmani": hizli_cikti(ol, mod, sure), "ham": {"yuz_bulunan_kare": ham["bulunan_kare"], "sure_sn": sure}}
    if not a.yalniz_olcum:
        sonuc["model"] = Analizci(cihaz=a.cihaz).yukle().analiz(kare, ses, sure, ol, mod, yunet=ok.yn)
    s = json.dumps(sonuc, ensure_ascii=False, indent=1, default=float)
    if a.cikti:
        open(a.cikti, "w", encoding="utf-8").write(s)
    print(s)


if __name__ == "__main__":
    main()

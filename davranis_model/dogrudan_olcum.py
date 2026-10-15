"""E1 DOGRUDAN OLCUM (v2.1, DURUST surum 2026-10-13): cikarimda MediaPipe Face Landmarker blendshape + librosa ses ozellikleri -> z -> katman (kanit/olasi) -> olcum cumlesi.
KATMANLAR (v21_plan.md §7.6): KANIT = yuz z>=2,6 / ses |z|>=2,25 VE nedensel kisi referansi yeterli (kisinin ONCEKI >=10 klibi ve >=30 sn); OLASI = E1 kalibre liste
(yuz z>=1,0, jawOpen 1,5, eyeWide 2,0; ses |z|>=1,5). Referans yetersizse z 'genel' (populasyon/MELD referansi) ve KANIT -> OLASI duser.
NEDENSELLIK: kisi referansi yalniz GECMIS kliplerden (KisiGecmisi.ekle'yi klip islendikten SONRA cagirin); sd tabani = 0,5 x populasyon sd (kucuk referansta z sismesin).
Yuz: klip basina 8 kare blendshape ORTALAMASI (52); sinyal = ilgili sutunlarin z ortalamasi. Ses: [rms_ortalama, sessizlik_orani, perde_ortalama_hz, perde_std_hz]."""
import numpy as np

try:
    from .olcum_dili import KANIT_ADAY, KANIT_SES_ESIK, KANIT_YUZ_ESIK, OLASI_SES_ESIK, OLASI_YUZ_ESIK, olcum_cumlesi
except ImportError:                                          # dogrudan calistirma/test
    from olcum_dili import KANIT_ADAY, KANIT_SES_ESIK, KANIT_YUZ_ESIK, OLASI_SES_ESIK, OLASI_YUZ_ESIK, olcum_cumlesi

YUZ_SUTUN = {"browDown": ["browDownLeft", "browDownRight"], "browOuterUp": ["browOuterUpLeft", "browOuterUpRight"], "cheekSquint": ["cheekSquintLeft", "cheekSquintRight"],
             "eyeSquint": ["eyeSquintLeft", "eyeSquintRight"], "eyeWide": ["eyeWideLeft", "eyeWideRight"], "jawOpen": ["jawOpen"],
             "mouthSmile": ["mouthSmileLeft", "mouthSmileRight"], "mouthStretch": ["mouthStretchLeft", "mouthStretchRight"], "mouthUpperUp": ["mouthUpperUpLeft", "mouthUpperUpRight"]}
SES_AD = ["rms_ortalama", "sessizlik_orani", "perde_ortalama_hz", "perde_std_hz"]
REF_MIN_KLIP = 10
REF_MIN_SN = 30.0
SD_TABAN = 0.5


def _z(x, ort, std, taban=None):
    std = np.asarray(std, dtype=np.float64)
    if taban is not None:
        std = np.maximum(std, SD_TABAN * np.asarray(taban, dtype=np.float64))
    return (np.asarray(x, dtype=np.float64) - np.asarray(ort, dtype=np.float64)) / np.where(std > 1e-9, std, 1.0)


def referans_olustur(yuz, ses, isim, kaynak=""):
    """POPULASYON referansi (yuz [N,52] NaN olabilir, ses [N,4], isim: 52 blendshape adi) -> JSON'a yazilabilir sozluk. Kisi referansi burada YOK (nedensel: KisiGecmisi)."""
    yuz, ses = np.asarray(yuz, float), np.asarray(ses, float)
    return {"kaynak": kaynak, "yuz_isim": list(isim), "ses_ad": SES_AD, "n": int(len(yuz)), "yuz_ort": np.nanmean(yuz, 0).tolist(), "yuz_std": np.nanstd(yuz, 0).tolist(),
            "ses_ort": np.nanmean(ses, 0).tolist(), "ses_std": np.nanstd(ses, 0).tolist()}


class KisiGecmisi:
    """Bir konusmacinin ONCEKI klipleri (akan kayit). Kullanim: z/isaretler hesaplandiktan SONRA ekle(yuz, ses, sure_sn) cagrilir (gelecek sizintisi yok)."""

    def __init__(self):
        self.yuz, self.ses, self.sure = [], [], 0.0

    def ekle(self, yuz, ses, sure_sn):
        self.yuz.append(np.full(52, np.nan) if yuz is None else np.asarray(yuz, float))
        self.ses.append(np.full(4, np.nan) if ses is None else np.asarray(ses, float))
        self.sure += float(sure_sn or 0.0)

    def yeterli(self):
        return len(self.yuz) >= REF_MIN_KLIP and self.sure >= REF_MIN_SN

    def referans(self):
        with np.errstate(all="ignore"):
            Y, S = np.array(self.yuz), np.array(self.ses)
            return {"yuz_ort": np.nanmean(Y, 0), "yuz_std": np.nanstd(Y, 0), "ses_ort": np.nanmean(S, 0), "ses_std": np.nanstd(S, 0)}


def sinyal_z(yuz_ham, ses_ham, pop, gecmis=None):
    """-> ({yuz_sinyali: z}, {ses_ozelligi: z}, baglam 'kisi'|'genel'). baglam 'kisi' yalniz gecmis yeterliyse."""
    isim = pop["yuz_isim"]
    baglam = "kisi" if (gecmis is not None and gecmis.yeterli()) else "genel"
    r = gecmis.referans() if baglam == "kisi" else pop
    out_y, out_s = {}, {}
    with np.errstate(all="ignore"):
        if yuz_ham is not None and not np.isnan(np.asarray(yuz_ham, float)).all():
            zc = _z(yuz_ham, r["yuz_ort"], r["yuz_std"], pop["yuz_std"] if baglam == "kisi" else None)
            for s, cols in YUZ_SUTUN.items():
                v = float(np.nanmean([zc[isim.index(c)] for c in cols]))
                out_y[s] = None if np.isnan(v) else v
        if ses_ham is not None and not np.isnan(np.asarray(ses_ham, float)).all():
            zs = _z(ses_ham, r["ses_ort"], r["ses_std"], pop["ses_std"] if baglam == "kisi" else None)
            out_s = {a: (None if np.isnan(zs[i]) else float(zs[i])) for i, a in enumerate(SES_AD)}
    return out_y, out_s, baglam


def katman_ata(zy, zs, baglam):
    """z sozluklerinden katmanli isaret listesi (KANIT once). Her oge: sinyal, z, yon, katman, baglam."""
    sonuc = []
    for s, z in zy.items():
        if z is None or z < OLASI_YUZ_ESIK.get(s, 99.0):
            continue
        kanit = baglam == "kisi" and s in KANIT_ADAY and z >= KANIT_YUZ_ESIK
        sonuc.append({"sinyal": s, "z": round(z, 3), "yon": 1, "katman": "kanit" if kanit else "olasi", "baglam": baglam})
    for a, z in zs.items():
        if z is None or abs(z) < OLASI_SES_ESIK:
            continue
        kanit = baglam == "kisi" and abs(z) >= KANIT_SES_ESIK
        sonuc.append({"sinyal": a, "z": round(z, 3), "yon": 1 if z > 0 else -1, "katman": "kanit" if kanit else "olasi", "baglam": baglam})
    sonuc.sort(key=lambda d: (d["katman"] != "kanit", -abs(d["z"])))
    return sonuc


def isaretler(yuz_ham, ses_ham, pop, gecmis=None):
    """Tam cikti: katmanli isaretler + ölçüm cümleleri. NOT: bu cagridan SONRA gecmis.ekle(...) yapin."""
    zy, zs, baglam = sinyal_z(yuz_ham, ses_ham, pop, gecmis)
    r = katman_ata(zy, zs, baglam)
    for x in r:
        x["cumle"] = olcum_cumlesi(x["sinyal"], x["z"], x["yon"], baglam, x["katman"])
    return r


# ---------------------------------------------------------------- olcum cikarma (mediapipe/librosa gerekir; CPU)
def yuz_ortalama(kareler_rgb, landmarker, kirp=None):
    """kareler_rgb: [K] uint8 HxWx3 RGB. kirp(frame)->frame (YuNet kutu kirpma) opsiyonel; None donerse kare atlanir (yedek merkez kirpma YOK).
    Donus: (52 ortalama | None, isim, bulunan_kare)."""
    import mediapipe as mp
    sk, isim = [], None
    for fr in kareler_rgb:
        f = kirp(fr) if kirp else fr
        if f is None:
            continue
        r = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        if r.face_blendshapes:
            isim = [c.category_name for c in r.face_blendshapes[0]]
            sk.append([c.score for c in r.face_blendshapes[0]])
    return (np.mean(sk, 0) if sk else None), isim, len(sk)


def ses_ozellik(y, sr=16000):
    """-> [rms_ortalama, sessizlik_orani, perde_ortalama_hz, perde_std_hz] (e1oran/a_ses.py ile AYNI)."""
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 3200:
        return [np.nan] * 4
    rms = librosa.feature.rms(y=y)[0]
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr)
    f0g = f0[f0 > 0]
    return [float(rms.mean()), float((rms < 0.01).mean()), float(f0g.mean()) if len(f0g) else np.nan, float(f0g.std()) if len(f0g) else np.nan]

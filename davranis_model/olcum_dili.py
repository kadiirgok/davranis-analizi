"""v2.1 OLCUM DILI (kullanici karari 2026-10-12): yuz/ses iddialari kullaniciya ALGI iddiasi degil OLCUM olarak sunulur.
  YANLIS (algi):  "kaşlarını çattı"           DOGRU (olcum): "ölçümde kaş indirme belirgin, z=1,8 (kişinin kendi ortalamasına göre)"
Bu modul YALNIZ metin uretir; hangi sinyalin hangi katmanda kullanilacagi E1 kalibrasyon listesindedir (bench_v2_5.txt §3). Davranis kernel'lerine (v2 bench/test/egitim) BAGLI DEGIL:
v2.1 E3 sablonu ile baglanacak (GPU kodu degisikligi = once CPU kontrolu).
Kurallar: (1) 'ölçümde' ile baslar; (2) z her zaman gosterilir (virgullu ondalik, 1 hane); (3) fiil/kisi/duygu yok (cattı, gülümsedi, kızgın vb. YASAK);
(4) kapali/koşullu sinyal icin metin URETILMEZ (None); (5) baglam sozu: z kisiye gore ise 'kişinin kendi önceki ölçümlerine göre' (NEDENSEL), populasyon z ise 'genel ortalamaya göre'."""

OLCUM_ADI = {
    "browDown": "kaş indirme", "browOuterUp": "kaş dış ucunu yukarı kaldırma", "cheekSquint": "yanak yukarı kalkması", "eyeSquint": "göz kısma",
    "eyeWide": "göz açıklığında artış", "jawOpen": "çene açıklığı", "mouthSmile": "ağız köşelerini yukarı çekme", "mouthStretch": "dudakları yana germe",
    "mouthUpperUp": "üst dudak yukarı kalkması",
    "rms_ortalama+": "ses seviyesi yüksek", "rms_ortalama-": "ses seviyesi düşük", "sessizlik_orani+": "duraksama payı yüksek", "sessizlik_orani-": "duraksama payı düşük",
    "perde_ortalama_hz+": "ses perdesi yüksek", "perde_ortalama_hz-": "ses perdesi düşük", "perde_std_hz+": "perde değişkenliği yüksek", "perde_std_hz-": "perde değişkenliği düşük",
}
# E1 "DURUST" KATMANLAR (kullanici karari 2026-10-13; v21_plan.md §7.6):
#   KANIT = secici esikler: yuz z>=2,6, ses |z|>=2,25 (modalite basina atesleme <=%10; yargic >=0,85) VE nedensel kisi referansi yeterli (onceki >=10 klip ve >=30 sn);
#   OLASI = E1 kalibre listesi: yuz z>=1,0 (jawOpen 1,5; eyeWide 2,0), ses |z|>=1,5. Referans yetersizse 'genel' z ve KANIT -> OLASI duser.
KANIT_YUZ_ESIK = 2.6
KANIT_SES_ESIK = 2.25
OLASI_YUZ_ESIK = {"browDown": 1.0, "eyeSquint": 1.0, "mouthSmile": 1.0, "mouthStretch": 1.0, "mouthUpperUp": 1.0, "jawOpen": 1.5,
                  "browOuterUp": 1.0, "cheekSquint": 1.0, "eyeWide": 2.0}
OLASI_SES_ESIK = 1.5
KANIT_ADAY = ("browDown", "eyeSquint", "mouthSmile", "mouthStretch", "mouthUpperUp", "jawOpen")   # kanit katmanina cikabilen yuz sinyalleri (digerleri en fazla olasi)
KATMAN = {s: ("kanit" if s in KANIT_ADAY else "olasi", e) for s, e in OLASI_YUZ_ESIK.items()}      # (en yuksek katman, OLASI esigi) - geriye uyum
YASAK_SOZCUKLER = ("çattı", "çatıyor", "gülümsedi", "gülümsüyor", "kızgın", "üzgün", "korku", "öfke", "mutlu", "sinirli", "şaşırdı", "kaçırıyor", "baktı", "ağladı")


def _ondalik(z):
    return f"{z:.1f}".replace(".", ",")


def olcum_cumlesi(sinyal, z, yon=1, baglam="kisi", katman=None):
    """sinyal: grup adi (ses icin yon '+'/'-' yon argumanindan); z: olculen z; katman 'kanit'|'olasi' (None: esiklerden belirlenir; nedensel referans bilgisi YOKSA 'olasi' varsayilir).
    Dondurur str ya da None (esik alti / kapali / kosullu)."""
    ses = sinyal in ("rms_ortalama", "sessizlik_orani", "perde_ortalama_hz", "perde_std_hz")
    anahtar = sinyal + ("+" if yon >= 0 else "-") if ses else sinyal
    if ses:
        if (z > 0) != (yon >= 0) or abs(z) < OLASI_SES_ESIK:
            return None
        esik_k = KANIT_SES_ESIK
    else:
        if sinyal not in OLASI_YUZ_ESIK or z < OLASI_YUZ_ESIK[sinyal]:
            return None
        esik_k = KANIT_YUZ_ESIK if sinyal in KANIT_ADAY else 99.0
    if katman is None:
        katman = "kanit" if (baglam == "kisi" and abs(z) >= esik_k) else "olasi"
    ad = OLCUM_ADI[anahtar]
    ref = "kişinin kendi önceki ölçümlerine göre" if baglam == "kisi" else "genel ortalamaya göre"
    derece = "belirgin" if katman == "kanit" else "olası (düşük güvenli)"
    if ses:
        s = f"ölçümde {ad}, z={_ondalik(abs(z))} ({ref})" + ("" if katman == "kanit" else f", {derece}")
    else:
        s = f"ölçümde {ad} {derece}, z={_ondalik(abs(z))} ({ref})"
    assert not any(w in s for w in YASAK_SOZCUKLER)
    return s


if __name__ == "__main__":
    for a, z, y, b, kt in [("browDown", 2.8, 1, "kisi", None), ("browDown", 2.8, 1, "genel", None), ("eyeWide", 2.2, 1, "kisi", None), ("eyeWide", 1.4, 1, "kisi", None),
                           ("noseSneer", 3.0, 1, "kisi", None), ("bas_yaw", 2.0, 1, "kisi", None), ("rms_ortalama", 2.4, 1, "kisi", None), ("perde_std_hz", -1.9, -1, "kisi", None)]:
        print(a, z, b, "->", olcum_cumlesi(a, z, y, b, kt))

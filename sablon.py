"""
Cevap sablonu (v2): ogrenci egitimi ve cevap_anahtari.py'nin ORTAK modulu (dil modeli YOK, tamamen kural/sablon).
Kaggle'da dataset (davranis-cevap-anahtari) icinde sablon.py olarak gelir; cevap_anahtari.py her calistirmada
kendini dataset klasorune kopyalar, boylece egitim ve cevap anahtari ayni sablonu kullanir.

Kavramlar
  modalite : "tam" | "sadece_goz" (ses yok) | "sadece_kulak" (yuz yok)
  satir    : algi satiri = [(grup, yon, buyukluk), ...]  (kullanici mesajindaki "Algılanan işaretler: ..." satiri)
  guven    : "yüksek" | "orta" | "düşük"   (prompt'ta verilir; hedef metin buna uyar)
Hedef metnin dayanaklari ve alternatif cumlesindeki isaret SADECE satir icinden secilir.
"""
import zlib

# grup -> (yukselir, duser); None = o yonde ifade kullanilmaz.  Sira = yardimci_hedefler.csv z sutunlarinin sirasi
IFADE = {
    "browDown": ("kaş indirme", None),
    "browInnerUp": ("kaş iç uçlarını yukarı çekme", None),
    "browOuterUp": ("kaş dış uçlarını yukarı kaldırma", None),
    "cheekSquint": ("yanak yukarı kalkması", None),
    "eyeSquint": ("göz kısma", None),
    "eyeWide": ("göz açıklığında artış", None),
    "jawOpen": ("çene açıklığı", None),
    "mouthFrown": ("ağız köşelerini aşağı çekme", None),
    "mouthFunnel": ("dudakları öne büzme", None),
    "mouthPucker": ("dudak büzme", None),
    "mouthPress": ("dudakları bastırma", None),
    "mouthShrugLower": ("alt dudağı yukarı itme", None),
    "mouthSmile": ("ağız köşelerini yukarı çekme", None),
    "mouthStretch": ("dudakları yana germe", None),
    "mouthUpperUp": ("üst dudak yukarı kalkması", None),
    "noseSneer": ("burun kırışması", None),
    "rms_ortalama": ("ses seviyesi yüksek", "ses seviyesi düşük"),
    "sessizlik_orani": ("duraksama payı yüksek", "duraksama payı düşük"),
    "perde_ortalama_hz": ("ses perdesi yüksek", "ses perdesi düşük"),
    "perde_std_hz": ("perde değişkenliği yüksek", "perde değişkenliği düşük"),
    # BAS/BAKIS: ogrenilmez, BlazeFace 6 noktadan GEOMETRIK hesaplanir (davranis_model/olcum_kurallari.py); z'leri ayri dosyada (bas_hedefler.csv)
    "bas_yaw": ("baş yönü sapması", None),
    "asagi_bakma": ("baş aşağı eğimi", None),
    "bakis_kacirma": ("bakış yönü sapması", None),
    "bas_geri_cekilme": ("baş geri çekilmesi", None),
}
GRUPLAR = list(IFADE)
SES_GRUPLARI = ("rms_ortalama", "sessizlik_orani", "perde_ortalama_hz", "perde_std_hz")
MODALITELER = ("tam", "sadece_goz", "sadece_kulak")
MAX_SATIR = 6          # algi satirindaki en fazla isaret
Z_ESIK = 1.0
MAX_DAYANAK = 4
ALT_IMZA_ILK = 3       # alternatif gerekcesi: rakip imzasinin en guclu ilk N isareti
GUVEN_SIRA = ["düşük", "orta", "yüksek"]
GUVEN_ESIK = (0.55, 0.80)   # kalibre edilmis en yuksek olasilik: <0.55 dusuk, <0.80 orta, digeri yuksek

ADJ = {"Ofke": "öfkeli", "Korku": "korkmuş", "Uzuntu": "üzgün", "Mutluluk": "mutlu", "Tiksinme": "tiksinmiş",
       "Saskinlik": "şaşkın", "Sakin": "sakin", "Notr": "nötr"}
ADJ_DE = {"Ofke": "öfkeli de", "Korku": "korkmuş da", "Uzuntu": "üzgün de", "Mutluluk": "mutlu da",
          "Tiksinme": "tiksinmiş de", "Saskinlik": "şaşkın da", "Sakin": "sakin de", "Notr": "nötr de"}
DUYGU_KOD = {"Ofke": "ofkeli", "Korku": "korkmus", "Uzuntu": "uzgun", "Mutluluk": "mutlu", "Tiksinme": "tiksinmis",
             "Saskinlik": "saskin", "Sakin": "sakin", "Notr": "notr"}

DURUM = {
    "yüksek": ["açıkça {a}", "net biçimde {a}", "belirgin şekilde {a}", "gayet {a}"],
    "orta": ["büyük ölçüde {a}", "çoğunlukla {a}", "{a} görünüyor", "{a} denebilir"],
    "düşük": ["kısmen {a}", "belki {a}", "biraz {a} gibi", "{a} olabilir, belirsiz"],
}
GUCLU_IFADE = ("açıkça", "net biçimde", "belirgin şekilde", "gayet")      # sadece yuksek guvende (durum alaninda)
DURUM_YOGUNLUK = {"yüksek": ("açıkça", "net biçimde", "belirgin şekilde", "gayet"),
                  "orta": ("büyük ölçüde", "çoğunlukla", "görünüyor", "denebilir"),
                  "düşük": ("kısmen", "belki", "biraz", "belirsiz")}
KUYRUK = {
    "yüksek": ["Bu izlenim oldukça net.", "Belirtiler tutarlı görünüyor.", "Bu yorum yerinde duruyor gibi.",
               "İfadeler birbirini güçlendiriyor."],
    "orta": ["Yine de bazı ayrıntılar tam oturmuyor.", "Bununla birlikte izlenim çok keskin değil.",
             "Yine de kesin konuşmak zor.", "Bazı ifadeler biraz belirsiz kalıyor."],
    "düşük": ["Ancak bu izlenim zayıf, temkinli olmak gerek.", "Yine de bu yorum oldukça belirsiz.",
              "Bu nedenle yorumu kesin saymamak gerekir.", "Bu nedenle kesin bir şey söylenemez."],
}
# {K}: kaynak ("Yüz ifadesinde ve seste" / "Yüz ifadesinde" / "Seste")
YORUM = {
    "destekliyor": [   # sadece DESTEKLEYEN isaretler sunulur
        "{K} şu işaretler dikkat çekiyor: {l}. Bu tablo kişinin {a} bir ruh halinde olduğuna işaret ediyor.",
        "Kişi {a} görünüyor: {l}. Bu ifadeler birbiriyle uyumlu bir bütün oluşturuyor.",
        "{K} şu işaretler göze çarpıyor: {l}. Bunlar {a} bir ruh haline işaret ediyor.",
        "Genel görünüm {a} bir kişiyi düşündürüyor. Bu izlenimde şu ifadeler öne çıkıyor: {l}.",
    ],
    "celisiyor": [     # destekleyen isaret yok; celisenler ({lc}) ayri cumleyle
        "Kişi {a} görünüyor ama bunu destekleyen bir işaret yok. Öte yandan bununla uyuşmayan ifade görülüyor: {lc}.",
        "{K} {a} bir ruh halini destekleyen işaret görülmüyor; ama bununla uyuşmayan ifade var: {lc}. Bu nedenle yorum temkinli tutulmalı.",
        "Bu izlenimi destekleyen bir ifade yok. Tersine, bununla uyuşmayan ifade dikkat çekiyor: {lc}.",
        "Kişi {a} olsa da dayanak bulunmuyor; bu izlenimle uyuşmayan ifade görülüyor: {lc}.",
    ],
}
# kismen: destekleyen ({ls}) ve celisen ({lc}) isaretler ayri ayri soylenir
YORUM_KISMEN = [
    "{Ls} {a} bir ruh haline işaret ediyor, ama bununla uyuşmayan ifade de görülüyor: {lc}.",
    "Kişi {a} görünüyor: {ls}. Ancak bu izlenimle uyuşmayan ifade de dikkat çekiyor: {lc}.",
    "{K} {ls} dikkat çekiyor ve bunlar {a} bir ruh haline uyuyor; fakat bununla uyuşmayan ifade de var: {lc}.",
    "{Ls} {a} bir ruh halini destekliyor. Öte yandan bununla uyuşmayan ifade de görülüyor: {lc}.",
]
YORUM_BOS = [
    "{K} belirgin bir hareket öne çıkmıyor. Kişi {a} olabilir ama bunu gösteren açık bir işaret yok.",
    "İfade oldukça durgun, ayırt edici bir işaret seçilemiyor. {A} bir ruh hali mümkün ama emin olmak zor.",
    "{K} ayırt edici bir şey öne çıkmıyor. {A} görünmesi mümkün, ancak dayanak zayıf.",
    "Öne çıkan bir ifade yok. {A} bir ruh haline dair güçlü bir belirti göremiyorum.",
]
ALT = [
    "{Ade} olabilir, çünkü şu işaret dikkat çekiyor: {c}.",
    "Bunu düşündüren işaret göze çarpıyor: {c}; bu nedenle {ade} olabilir.",
    "Kişi {ade} olabilir, çünkü şu işaret görülüyor: {c}.",
    "Ayrıca {ade} olabilir; çünkü şunu fark ediyorum: {c}.",
]
ALT_YOK = "belirgin bir alternatif yok"
YASAK = ["yüzde", "etiket", "model", "ölçüm", "olcum", "veri", "sayı", "%"]

KAYNAK = {"tam": "Yüz ifadesinde ve seste", "sadece_goz": "Yüz ifadesinde", "sadece_kulak": "Seste"}
# modalite dusurulmus ornekte yorum bununla baslar
YORUM_ON = {
    "sadece_goz": "Ses duyulmuyor, yalnızca yüz ifadesine göre değerlendirilebiliyor. ",
    "sadece_kulak": "Yüz görünmüyor, yalnızca sese göre değerlendirilebiliyor. ",
}
YORUM_ON_BASLANGIC = {"sadece_goz": "ses duyulmuyor, yalnızca yüz ifadesine göre",
                      "sadece_kulak": "yüz görünmüyor, yalnızca sese göre"}
GIRDI_SATIRI = {"sadece_goz": "Girdi: yalnızca görüntü, ses yok", "sadece_kulak": "Girdi: yalnızca ses, yüz yok"}

TERS_IFADE = {}
for _g, (_y, _d) in IFADE.items():
    for _f, _s in ((_y, 1), (_d, -1)):
        if _f is not None:
            assert _f not in TERS_IFADE, _f
            TERS_IFADE[_f] = (_g, _s)


def buyuk(s):
    if not s:
        return s
    return ("İ" if s[0] == "i" else s[0].upper()) + s[1:]


def birlestir(liste):
    if len(liste) <= 1:
        return "".join(liste)
    return ", ".join(liste[:-1]) + " ve " + liste[-1]


def sec(klip_id, tuz, n):
    return zlib.crc32(f"{klip_id}|{tuz}".encode("utf-8")) % n


def ifade_al(g, yn):
    return IFADE[g][0 if yn > 0 else 1]


def mevcut_gruplar(modalite):
    if modalite == "tam":
        return set(GRUPLAR)
    if modalite == "sadece_goz":
        return {g for g in GRUPLAR if g not in SES_GRUPLARI}
    if modalite == "sadece_kulak":
        return set(SES_GRUPLARI)
    raise ValueError(modalite)


# ------------------------------------------------------------------ guven

def guven_seviyesi(q):
    """Kalibre edilmis en yuksek sinif olasiligi -> yuksek/orta/dusuk."""
    return "yüksek" if q >= GUVEN_ESIK[1] else ("orta" if q >= GUVEN_ESIK[0] else "düşük")


def guven_dusur(g):
    """Bir kademe dusur (modalite dusurulmus girdi)."""
    return GUVEN_SIRA[max(GUVEN_SIRA.index(g) - 1, 0)]


ORT_Z_ESIK = 0.15      # etiketin o ozellikteki |ortalama z|'si bunun altindaysa "o yonde egilim yok" sayilir


class Imza(dict):
    """{duygu: [(grup, yon), ...] guce gore sirali} + .ort_z = {duygu: {grup: train ortalama z}} (tum 20 ozellik)."""
    ort_z = None


def imza_yukle(obj):
    """imza.json icerigi -> Imza. '_ort_z' anahtari opsiyonel (eski dosyalarda yok -> celisen yalniz ters-yon kuralina doner)."""
    im = Imza({d: [(g, int(yn)) for g, yn in sira] for d, sira in obj.items() if d != "_ort_z"})
    im.ort_z = obj.get("_ort_z")
    return im


def imza_json(imza):
    """Imza -> imza.json icerigi (yukleme ile ayni bicim)."""
    o = {d: [[g, int(yn)] for g, yn in k] for d, k in imza.items()}
    if getattr(imza, "ort_z", None):
        o["_ort_z"] = imza.ort_z
    return o


def uyum_siniflari(satir, sinif, imza):
    """Algi satirindaki isaretleri sinifin imzasina gore ayir: (destekleyen, celisen, notr); her biri [(grup, yon, mag)].
    destekleyen: (grup, yon) sinifin imzasinda. celisen: sinifin imzasinda olmayan VE baska bir duygunun imzasinin ILK
    ALT_IMZA_ILK isaretlerinden biri VE etiketin train ortalama z'si o yonle ters isaretli ya da |ort z| < ORT_Z_ESIK (etiket o yonde
    egilim gostermiyor; etiketin kendi ortalamasi ayni yondeyse - orn. uzuntude de gulumseme dusukse - celisen DEGIL, notr).
    notr: kalanlar (gerekce listesine girmez)."""
    sup = [t for t in satir if (t[0], t[1]) in imza[sinif]]
    baska_ilk = {tuple(x) for d, s_ in imza.items() if d != sinif for x in s_[:ALT_IMZA_ILK]}
    oz = (getattr(imza, "ort_z", None) or {}).get(sinif)

    def ters(t):
        if oz is None:                                # ort_z yoksa (eski imza.json): imzadaki isaretin ters yonu
            return (t[0], -t[1]) in imza[sinif]
        m = float(oz[t[0]])
        return m * t[1] < 0 or abs(m) < ORT_Z_ESIK

    con = [t for t in satir if t not in sup and (t[0], t[1]) in baska_ilk and ters(t)]
    notr = [t for t in satir if t not in sup and t not in con]
    return sup, con, notr


def uyum_hesapla(satir, sinif, imza):
    """Algi satirinin sinif imzasiyla uyumu: destekliyor (destekleyen >= 1 VE destekleyen > celisen sayisi) / celisiyor (sadece celisen var,
    destekleyen yok) / kismen (esitlik, destekleyen <= celisen, ya da hicbiri yok). Egitimde gercek etiketle, cikarimda tahmin edilen
    sinifla AYNI fonksiyon kullanilir."""
    sup, con, _ = uyum_siniflari(satir, sinif, imza)
    if sup and len(sup) > len(con):
        return "destekliyor"
    if con and not sup:
        return "celisiyor"
    return "kismen"


UYUM_TAVANI = {"destekliyor": "yüksek", "kismen": "orta", "celisiyor": "düşük"}


def son_guven(satir, sinif, imza, q, modalite="tam"):
    """SON guven = min(olasilik tabanli guven, uyum tavani); modalite eksikse ayrica bir kademe dusurulur.
    q: (kalibre) en yuksek sinif olasiligi. Dondurur (guven, uyum)."""
    uyum = uyum_hesapla(satir, sinif, imza)
    g = min(guven_seviyesi(q), UYUM_TAVANI[uyum], key=GUVEN_SIRA.index)
    if modalite != "tam":
        g = guven_dusur(g)
    return g, uyum


# ------------------------------------------------------------------ algi satiri

def satir_cikar(z, modalite="tam", esik=Z_ESIK, max_n=MAX_SATIR):
    """z: GRUPLAR sirasinda 20 (ya da bas isaretleri dahil 24) z-skoru (gercek ya da yardimci baslik tahmini) -> [(grup, yon, |z|)] buyukten kucuge."""
    var = mevcut_gruplar(modalite)
    cikti = []
    for k, g in enumerate(GRUPLAR):
        if g not in var or k >= len(z):                 # z 20 (eski) ya da 24 (bas isaretleri dahil) uzunlukta olabilir
            continue
        v = float(z[k])
        yn = 1 if v > 0 else -1
        if abs(v) >= esik and ifade_al(g, yn) is not None:
            cikti.append((g, yn, abs(v)))
    cikti.sort(key=lambda t: -t[2])
    return cikti[:max_n]


def satir_gurultu(satir, z, modalite, rng, p_dus=0.20, p_ekle=0.10, max_n=MAX_SATIR):
    """Yardimci basligin hatalarini taklit: %p_dus bir isaret dusur, %p_ekle klipte olmayan (|z|<1) bir isaret ekle."""
    satir = list(satir)
    if satir and rng.random() < p_dus:
        satir.pop(rng.randint(len(satir)))
    if rng.random() < p_ekle:
        var = mevcut_gruplar(modalite)
        mevcut = {g for g, _, _ in satir}
        adaylar = [(g, yn) for k, g in enumerate(GRUPLAR) if g in var and g not in mevcut and k < len(z) and abs(float(z[k])) < Z_ESIK
                   for yn in (1, -1) if ifade_al(g, yn) is not None]
        if adaylar:
            g, yn = adaylar[rng.randint(len(adaylar))]
            satir.append((g, yn, float(rng.uniform(1.0, 1.6))))
    satir.sort(key=lambda t: -t[2])
    return satir[:max_n]


def kullanici_eki(satir, guven, modalite="tam"):
    """Kullanici mesajina (algi token'larindan sonra) eklenen satirlar."""
    ifadeler = [ifade_al(g, yn) for g, yn, _ in satir]
    parcalar = []
    if modalite in GIRDI_SATIRI:
        parcalar.append(GIRDI_SATIRI[modalite])
    parcalar.append("Algılanan işaretler: " + ("; ".join(ifadeler) if ifadeler else "belirgin işaret yok"))
    parcalar.append(f"Güven: {guven}")
    return "\n".join(parcalar)


# ------------------------------------------------------------------ hedef metin

def hedef_uret(klip_id, etiket, olas, imza, satir, modalite, guven):
    """Cevap sozlugu (alan sirasi: dayanaklar, duygu, durum, yorum, alternatif, guven; ayrica ipucu_uyumu).
    olas: {sinif: olasilik} (sadece alternatifteki rakip sinif icin); satir: algi satiri; guven: SON guven (son_guven
    ile hesaplanmis). Dayanaklar ve alternatifteki isaret SADECE satirdan secilir."""
    siniflar = sorted(olas, key=lambda c: -olas[c])
    rakip = next(c for c in siniflar if c != etiket)
    p_rakip = float(olas[rakip])
    uyum = uyum_hesapla(satir, etiket, imza)
    sup, con, notr = uyum_siniflari(satir, etiket, imza)
    ifd = lambda ts: [ifade_al(g, yn) for g, yn, _ in ts]

    a, K = ADJ[etiket], KAYNAK[modalite]
    durum = DURUM[guven][sec(klip_id, "durum", 4)].format(a=a)
    i = sec(klip_id, "yorum", 4)
    if sup and con:                                   # destekleyen ve celisen birlikte (uyum destekliyor ya da kismen): celisenler ayri cumlede
        secilen = sup[:3]
        Ls, Lc = birlestir(ifd(sup[:3])), birlestir(ifd(con[:2]))
        govde = YORUM_KISMEN[i].format(a=a, A=buyuk(a), ls=Ls, Ls=buyuk(Ls), lc=Lc, K=K)
    elif sup:
        secilen = sup[:MAX_DAYANAK]
        L = birlestir(ifd(secilen))
        govde = YORUM["destekliyor"][i].format(a=a, A=buyuk(a), l=L, L=buyuk(L), K=K)
    elif con:
        secilen = []                                  # gerekce listesi bos: celisenler dayanak degildir
        Lc = birlestir(ifd(con[:3]))
        govde = YORUM["celisiyor"][i].format(a=a, A=buyuk(a), lc=Lc, K=K)
    else:                                             # destekleyen de celisen de yok (bos / notr): notr isaretler listelenmez
        secilen = []
        govde = YORUM_BOS[i].format(a=a, A=buyuk(a), K=K)
    dayanaklar = ifd(secilen)
    yorum = YORUM_ON.get(modalite, "") + govde + " " + KUYRUK[guven][sec(klip_id, "kuyruk", 4)]

    alternatif = ALT_YOK
    if p_rakip > 0.2 or uyum == "celisiyor":
        # gerekce: algi satirindaki isaretlerden, rakip duygunun imzasinin EN GUCLU ilk 3 isareti (imza[rakip] guce gore
        # sirali) ve yonu imzayla ayni olanlar; birden fazlaysa klipteki |z|'si en buyuk. Yoksa alternatif yok.
        ilk3 = [tuple(x) for x in imza[rakip][:ALT_IMZA_ILK]]
        aday_r = [(g, yn, m) for g, yn, m in satir if (g, yn) in ilk3]
        if aday_r:
            g_, yn_, _ = max(aday_r, key=lambda t: t[2])
            c = ifade_al(g_, yn_)
            ade = ADJ_DE[rakip]
            alternatif = ALT[sec(klip_id, "alt", 4)].format(Ade=buyuk(ade), ade=ade, a=ADJ[rakip], c=c, C=buyuk(c))

    return {"ipucu_uyumu": uyum, "duygu": DUYGU_KOD[etiket], "durum": durum, "yorum": yorum,
            "dayanaklar": dayanaklar, "alternatif": alternatif, "guven": guven}

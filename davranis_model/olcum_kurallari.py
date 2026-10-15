"""Kural tabanli (ogrenilmeyen) olcum islevleri. Bagimlilik: yalniz numpy. Model forward'i (modeling_davranis), hedef_v2 (egitim etiketi),
asama0 ve ogretmen AYNI islevleri kullanir.

1) BAS/BAKIS isaretleri: BlazeFace 6 nokta (kirpinti icinde 0..1; sira: sag goz, sol goz, burun, agiz, sag kulak, sol kulak) -> GEOMETRIK baş yönü.
   OGRENILMEZ (olcum basligina ogretilmez); z, KISI BAZLI normalizasyonla hesaplanir. Dereceler yaklasiktir (kalibre degil; kureye benzetim
   r ~ 0.5*tan(aci)); "sozde" olcumdur: gercek bas pozu etiketi (MediaPipe donusum matrisi) veri-hazirlikte KAYDEDILMEDI.
   bas_geri_cekilme: yuz KUTUSUNUN kare icinde kuculmesi gerekir; onbellekte kutu boyutu YOK (kirpinti normalizasyonu bunu siler) ->
   kutu_olcek verilmezse isaret yok sayilir (z=0, mevcut=False).
2) TEKRAR EDEN KALIPLAR: ayni olculmus isaretin >=2 parcada gorulmesi -> {"kalip": "<ifade> (<baglam>)", "sayi": n}.
"""
import math

import numpy as np

BAS_GRUPLARI = ("bas_yaw", "asagi_bakma", "bakis_kacirma", "bas_geri_cekilme")
BAS_IFADE = {
    "bas_yaw": ("başı yana dönük", None),
    "asagi_bakma": ("aşağı bakıyor", None),
    "bakis_kacirma": ("bakışını kaçırıyor", None),
    "bas_geri_cekilme": ("başını geri çekiyor", None),
}
ACI_ESIK = 20.0                 # bakis_kacirma: baş yönü kameradan bu dereceden fazla
GERI_MIN_ORAN = 0.04           # bas_geri_cekilme: yuz kutusu genisligi ilk yariya gore en az %4 kuculmeli (8882 klibin %99'u ~%5.4 altinda; daha azi gurultu)
KACIRMA_MIN_PAY = 0.25          # bakis_kacirma: karelerin en az bu payi esigi asmali (8 karede >=2)
MIN_YUZ_KARE = 4                # klipte yuz bulunan kare sayisi bundan azsa baş isaretleri yok sayilir (yedek medyan noktalarla dolu)
KISI_MIN_KLIP = 6               # kisi bazli norm icin en az klip; altinda genel norm
Z_KIRP = 3.0
# 8882 klibin genel (ort, sd): kisi tablosu verilmediginde (tek klip/parca cikarimi) kullanilir. Sira BAS_GRUPLARI; geri_cekilme icin veri yok.
GENEL_VARSAYILAN = {"genel": [[4.92822, 3.86614], [3.66317, 4.75114], [0.04462, 0.14870], [0.0, 1.0]]}
# --- E0 DUZELTMESI (v2.1-guven-2): BlazeFace yaw'i MediaPipe baş pozunun ~1,81 KATI abartiyor (858 test klibi: BF ~ 1.81*MP + 0.71, r=0.89). Z (kisi-bagil) HESABI DEGISMEZ (dondurulmus
# tablo ayni birimde); iddia metni 'mutlak' bir durum soyledigi icin z'ye MUTLAK (kalibre derece) KAPI eklenir: kapi altindaysa z <= 0 (isaret iddia edilmez).
ACI_OLCEK = 0.553               # BlazeFace derecesi * ACI_OLCEK ~ MediaPipe derecesi (1/1.81)
YAW_MIN_DERECE = 15.0           # bas_yaw: kalibre ort |yaw| bu degerin altindaysa 'başı yana dönük' denmez (v2'de ateslenen klipler medyan MP yaw 6.2 derece, hicbiri >=15)
PITCH_MIN_DERECE = 10.0         # asagi_bakma: kalibre ort asagi pitch bu degerin altindaysa 'aşağı bakıyor' denmez
KACIRMA_ACI = 20.0              # bakis_kacirma: baş yonu kameradan KALIBRE bu dereceden fazla olan kare payi KACIRMA_MIN_PAY'i asmali (ham ACI_ESIK BlazeFace biriminde kalir; z tablosu degismez)
PITCH_REF = 0.116               # onbellekteki (burun_y - kulak_orta_y)/kulak_mesafesi MEDYANI (8882 klip): bu deger 0 derece kabul edilir


def kare_acilari(kp):
    """kp [...,6,2] (kirpinti icinde 0..1) -> (yaw_derece, pitch_derece) [...]. pitch > 0 = asagi. Yaklasik (r ~ 0.5*tan(aci))."""
    kp = np.asarray(kp, np.float32)
    kr, kl, bu = kp[..., 4, :], kp[..., 5, :], kp[..., 2, :]
    ed = np.maximum(np.linalg.norm(kr - kl, axis=-1), 0.2)
    orta = (kr + kl) / 2
    r = (bu[..., 0] - orta[..., 0]) / ed
    p = (bu[..., 1] - orta[..., 1]) / ed - PITCH_REF
    return np.degrees(np.arctan(2.0 * r)), np.degrees(np.arctan(2.0 * p))


def klip_ham(geo, yuz_bulunan=None, kutu_olcek=None):
    """geo [N,K,>=12] (ilk 12 = 6 nokta x,y) -> (ham [N,4], gecerli [N,4] bool). Sutunlar BAS_GRUPLARI sirasi.
    bas_yaw: ort |yaw| (derece); asagi_bakma: ort max(pitch,0) (derece); bakis_kacirma: baş yönü >ACI_ESIK olan kare payi;
    bas_geri_cekilme: (ilk yari ort. kutu_olcek - son yari ort.)/ilk yari, >=0 (kutu_olcek [N,K] = kutu genisligi/kare genisligi; None = yok)."""
    geo = np.asarray(geo, np.float32)
    N, K = geo.shape[:2]
    yaw, pitch = kare_acilari(geo[..., :12].reshape(N, K, 6, 2))
    ham = np.zeros((N, 4), np.float32)
    ham[:, 0] = np.abs(yaw).mean(1)
    ham[:, 1] = np.maximum(pitch, 0).mean(1)
    ham[:, 2] = ((np.abs(yaw) > ACI_ESIK) | (np.abs(pitch) > ACI_ESIK)).mean(1)
    gecerli = np.ones((N, 4), bool)
    if yuz_bulunan is not None:
        gecerli &= (np.asarray(yuz_bulunan) >= MIN_YUZ_KARE)[:, None]
    if kutu_olcek is None:
        gecerli[:, 3] = False
    else:
        ko = np.asarray(kutu_olcek, np.float32)
        y = K // 2
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ilk, son = np.nanmean(ko[:, :y], 1), np.nanmean(ko[:, y:], 1)
        ok = np.isfinite(ilk) & np.isfinite(son)                          # her yarida >=1 yuz bulunan kare
        ham[:, 3] = np.where(ok, np.maximum((ilk - son) / np.maximum(np.where(ok, ilk, 1.0), 1e-6), 0.0), 0.0)
        gecerli[:, 3] &= ok
    return ham, gecerli


def kisi_norm(ham, gecerli, kisi):
    """Kisi bazli (ort, sd) tablosu: {kisi: [4,2]} + 'genel': [4,2]. Kisi basina >=KISI_MIN_KLIP gecerli klip yoksa genel norm kullanilir.
    sd tabani: genel sd'nin %50'si (az degiskenli isaretlerde tek karelik sapmanin dev z uretmesini onler)."""
    ham, gecerli, kisi = np.asarray(ham, np.float64), np.asarray(gecerli), np.asarray(kisi)
    genel = np.zeros((4, 2))
    for j in range(4):
        v = ham[gecerli[:, j], j]
        genel[j] = (v.mean(), v.std()) if len(v) > 1 else (0.0, 1.0)
    tablo = {"genel": genel}
    for k in sorted(set(kisi.tolist())):
        m = kisi == k
        t = genel.copy()
        for j in range(4):
            v = ham[m & gecerli[:, j], j]
            if len(v) >= KISI_MIN_KLIP:
                t[j] = (v.mean(), max(v.std(), 0.5 * genel[j, 1]))
        tablo[k] = t
    return tablo


def bas_z(ham, gecerli, kisi, tablo):
    """ham [N,4] -> z [N,4] (kisi bazli; kisi tabloda yoksa genel), [-Z_KIRP, Z_KIRP]; gecersiz = 0. bakis_kacirma: kare payi KACIRMA_MIN_PAY altindaysa z<=0."""
    z = np.zeros_like(ham, dtype=np.float32)
    for i in range(len(ham)):
        t = tablo.get(str(kisi[i]), tablo["genel"])
        for j in range(4):
            if gecerli[i, j]:
                z[i, j] = (ham[i, j] - t[j, 0]) / max(t[j, 1], 1e-6)
    z[:, 2] = np.where(ham[:, 2] < KACIRMA_MIN_PAY, np.minimum(z[:, 2], 0.0), z[:, 2])
    z[:, 3] = np.where(ham[:, 3] < GERI_MIN_ORAN, np.minimum(z[:, 3], 0.0), z[:, 3])
    return np.clip(z, -Z_KIRP, Z_KIRP)


def bas_olcumleri(geo, yuz_bulunan, tablo, kisi=None, kutu_olcek=None):
    """MODEL FORWARD'INDA kullanilan tek giris noktasi: geo [N,K,>=12] -> (z [N,4], mevcut [N,4]). kisi None = genel norm."""
    ham, gec = klip_ham(geo, yuz_bulunan, kutu_olcek)
    kisi = np.array(["genel"] * len(ham)) if kisi is None else np.asarray(kisi)
    z = bas_z(ham, gec, kisi, tablo)
    # MUTLAK KAPILAR (kalibre derece): kisi-bagil z kucuk acilari 'yana dönük/aşağı bakıyor' diye adlandirmasin
    g_ = np.asarray(geo, np.float32)
    yaw, pitch = kare_acilari(g_[..., :12].reshape(g_.shape[0], g_.shape[1], 6, 2))
    yaw, pitch = yaw * ACI_OLCEK, pitch * ACI_OLCEK
    z[:, 0] = np.where(np.abs(yaw).mean(1) >= YAW_MIN_DERECE, z[:, 0], np.minimum(z[:, 0], 0.0))
    z[:, 1] = np.where(np.maximum(pitch, 0).mean(1) >= PITCH_MIN_DERECE, z[:, 1], np.minimum(z[:, 1], 0.0))
    kac = ((np.abs(yaw) > KACIRMA_ACI) | (np.abs(pitch) > KACIRMA_ACI)).mean(1)
    z[:, 2] = np.where(kac >= KACIRMA_MIN_PAY, z[:, 2], np.minimum(z[:, 2], 0.0))
    return z, gec


def tablo_json(tablo):
    return {k: np.asarray(v).round(5).tolist() for k, v in tablo.items()}


def tablo_yukle(obj):
    return {k: np.asarray(v, np.float64) for k, v in obj.items()}


# ------------------------------------------------------------------------------------------ tekrar eden kaliplar
KALIP_MAX = 3


def tekrar_kaliplari(olcum_parca, uyum_parca=None, konusma_parca=None):
    """olcum_parca: parca basina olculmus isaret ifadeleri (liste/liste); uyum_parca: parca basina 'uyumlu'|'kismen'|'uyumsuz' (None = bilinmiyor);
    konusma_parca: parca basina bool (transkript var). Ayni ifade >=2 FARKLI parcada gorulduyse -> [{"kalip": "<ifade> (<baglam>)", "sayi": n}].
    Baglam: isaretin gorulduğu TUM parcalar uyumsuzsa 'uyumsuz parçalarda'; degilse parcalarin cogunlugu (>= yarisi) konusmaliysa 'konuşurken',
    degilse 'susarken'. konusma_parca None ise baglam 'uyumsuz parçalarda' ya da (belirsiz) 'konuşurken'. Siralama: sayi azalan, ifade alfabetik; en fazla KALIP_MAX."""
    P = len(olcum_parca)
    parcalar = {}
    for pi, ol in enumerate(olcum_parca):
        for ifade in dict.fromkeys(ol):                       # parca basina bir kez
            if ifade and ifade != "ölçülmüş belirgin işaret yok":
                parcalar.setdefault(ifade, []).append(pi)
    cikti = []
    for ifade, idx in parcalar.items():
        if len(idx) < 2:
            continue
        if uyum_parca is not None and all(uyum_parca[i] == "uyumsuz" for i in idx):
            baglam = "uyumsuz parçalarda"
        elif konusma_parca is not None and sum(bool(konusma_parca[i]) for i in idx) * 2 < len(idx):
            baglam = "susarken"
        else:
            baglam = "konuşurken"
        cikti.append({"kalip": f"{ifade} ({baglam})", "sayi": len(idx)})
    cikti.sort(key=lambda k: (-k["sayi"], k["kalip"]))
    return cikti[:KALIP_MAX]


def kalip_dizisi_metni(kaliplar):
    """Forward'da ZORLANAN 'tekrar_eden_kaliplar' dizisi (kompakt JSON). hedef_v2 (egitim etiketi) ve model AYNI islevi kullanir."""
    import json
    return json.dumps(kaliplar, ensure_ascii=False, separators=(",", ":"))

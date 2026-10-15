"""v2.1.1 hedef/sablon katmani (cikarim kopyasi): SablonUretici, ruh hali/yorum yardimcilari. Egitim veri seti cumleleri ve duman bankasi bu kopyada YOKTUR.

sablon.py + imza.json (v1/v2 kural sablonu) + cumle bankasi -> {zaman_cizelgesi:[...], olay:{...}} hedef JSON'u.
Gercek etiketler Kaggle'da 7B ogretmenden (kaggle-ogretmen) gelecek; bu modul (a) duman testi icin, (b) ogretmen
hedefi yokken yedek/sablon etiket olarak kullanilir. Alan adlari ve sema ogretmenle AYNIDIR (modeling_davranis.sema_gecerli).

Transkript icerigi (hedef A): %40 orijinal (CREMA-D/RAVDESS sabit notr cumle), %30 klibin duygusuyla UYUSAN banka cumlesi
("uyumlu"), %30 UYUSMAYAN banka cumlesi ("uyumsuz"). Zaman cizelgesi ornekleri AYNI konusmacinin 3-5 klibinden kurulur (hedef C).
"""
import importlib.util
import json
import os
import random
import re
from collections import Counter

KOK = os.environ.get("DAVRANIS_KOK", os.path.dirname(os.path.abspath(__file__)))   # sablon.py + imza.json klasoru


def _kural_yukle():
    """davranis_model/olcum_kurallari.py (bagimlilik: numpy; model forward'i ile AYNI islevler): hedef_v2'nin yaninda ya da KOK altinda."""
    for d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "davranis_model"), os.path.join(KOK, "davranis_model")):
        yol = os.path.join(d, "olcum_kurallari.py")
        if os.path.exists(yol):
            spec = importlib.util.spec_from_file_location("olcum_kurallari", yol)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("davranis_model/olcum_kurallari.py bulunamadi")


KURAL = _kural_yukle()


def _ak_yukle():
    """davranis_model/anlam_kurallari.py: anlam cozumleme + UYUM + birincil/ikincil anlam + ruh_hali/tutum + parca yorumu (egitim hedefi, ogretmen ve model AYNI islevleri kullanir)."""
    for d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "davranis_model"), os.path.join(KOK, "davranis_model")):
        yol = os.path.join(d, "anlam_kurallari.py")
        if os.path.exists(yol):
            spec = importlib.util.spec_from_file_location("anlam_kurallari", yol)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("davranis_model/anlam_kurallari.py bulunamadi")


AK = _ak_yukle()

ISARET_A, ISARET_B = chr(1), chr(2)       # modeling_davranis.HEDEF_ISARET ile ayni
ISARET_ZA, ISARET_ZB = chr(3), chr(4)     # modeling_davranis.ZORLA_ISARET ile ayni (modelin forward'inda kopyalanan kanit dizisi)
YOK_IFADE = "ölçülmüş belirgin işaret yok"
NOTR_DUYGU = ("Notr", "Sakin")

DUYGU_ANAHTAR = {"Ofke": "ofkeli", "Korku": "korkmus", "Uzuntu": "uzgun", "Mutluluk": "mutlu", "Tiksinme": "tiksinmis",
                 "Saskinlik": "saskin", "Sakin": "sakin", "Notr": "notr"}
TUTUM = {"Ofke": "baskin", "Korku": "kacingan", "Uzuntu": "geri_cekilmis", "Mutluluk": "kendinden_emin",
         "Tiksinme": "savunmaci", "Saskinlik": "kacingan", "Sakin": "kendinden_emin", "Notr": "geri_cekilmis"}
MOD_GIRDI = {"tam": "goruntu+ses", "sadece_goz": "sadece_goruntu", "sadece_kulak": "sadece_ses"}


def sablon_yukle():
    spec = importlib.util.spec_from_file_location("sablon", os.path.join(KOK, "sablon.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(os.path.join(KOK, "imza.json"), encoding="utf-8") as f:
        imza = mod.imza_yukle(json.load(f))
    return mod, imza


SOZ_DUYGU_TR = {"Ofke": "öfke", "Korku": "korku", "Uzuntu": "üzüntü", "Mutluluk": "mutluluk", "Tiksinme": "tiksinme", "Saskinlik": "şaşkınlık"}
RUH_HALI_LISTE = ("sakin", "rahat", "neşeli", "heyecanlı", "gergin", "kaygılı", "öfkeli", "kırgın", "üzgün", "keyifsiz", "şaşkın", "tiksinmiş", "rahatsız",
                  "mesafeli", "kararsız", "değerlendirilemedi")
ISARETSIZ_ON = {"sadece_goz": "ses girdisi yok; ", "sadece_kulak": "görüntü girdisi yok; "}

# --- GUVEN KURALI (v2.1-guven-1; cikarim zamani, yeniden egitim YOK): olay.duygu icin sinif olasiligi GUVEN_ESIK[0] altindaysa duygu adi YAZILMAZ.
CIKARIM_YOK = "çıkarım yapılamıyor"
DUYGU_ADI_KOKLERI = ("öfke", "kızgın", "kızgınlık", "sinirli", "korku", "korkmuş", "korkulu", "endişe", "kaygı", "panik", "üzgün", "üzüntü", "hüzün", "keder",
                     "mutlu", "mutluluk", "neşe", "sevinç", "keyif", "tiksin", "iğren", "nefret", "şaşkın", "şaşır", "şaşkınlık", "nötr", "notr", "sakin")


def genel_duygu_adlari(metin):
    """Metindeki duygu adi/es anlamlilari (kok eslesmesi; kucuk harf). 'duygu' sozcugu tek basina ad SAYILMAZ. Dondurur sirali benzersiz liste."""
    t = (metin or "").replace("İ", "i").replace("I", "ı").lower()
    bulunan = set()
    for kok in DUYGU_ADI_KOKLERI:
        if re.search(r"(?<![\wçğıöşü])" + re.escape(kok), t):
            bulunan.add(kok)
    return sorted(bulunan)


def isaretsiz_yorum(p, olasi=()):
    """KODLA yazilan parca yorumu (LLM yazmaz): kanit da olasi isaret de yoksa sabit kalip. Ses yok / yuz yok onekleri korunur.
    olasi verilirse (yalniz sablon yedegi icin) 1. cumle olasi isareti temkinli anar."""
    on = ISARETSIZ_ON.get(p["mod"], "")
    if olasi:
        ilk = f"{p['t']} sn: {on}ölçüm katmanı eşiği geçen bir işaret saptamadı; ancak {olasi[0]} olabilir."
    else:
        ilk = f"{p['t']} sn: {on}ölçüm katmanı bu aralıkta eşiği geçen bir işaret saptamadı."
    duygu = SOZ_DUYGU_TR.get(p.get("soz_duygu"))
    son = (f"Söylenen içerik {duygu} duygusu taşıyor; ölçüm katmanı bunu destekleyen ya da çürüten bir işaret saptamadı."
           if duygu and p["mod"] != "sadece_goz" and (p.get("transkript") or "").strip() else "Söylenen içeriğin duygu sınıflandırması bir ipucu vermedi.")
    return f"{ilk} Bu nedenle ölçümlere dayalı bir duygu çıkarımı yapılamıyor. {son}"


def isaretsiz_genel(parcalar, duygu_adi_yok=False):
    """Tum parcalar isaretsizse genel_yorum (kodla). duygu_adi_yok=True (olay.duygu 'çıkarım yapılamıyor'): hicbir duygu adi gecmez."""
    emo = Counter(p.get("soz_duygu") for p in parcalar if p.get("soz_duygu") in SOZ_DUYGU_TR)
    modlar = {p["mod"] for p in parcalar}
    eksik = (" Bazı parçalarda ses girdisi yok." if "sadece_goz" in modlar else "") + (" Bazı parçalarda görüntü girdisi yok." if "sadece_kulak" in modlar else "")
    if duygu_adi_yok:
        soz = ("Söylenen içerikte duygusal bir ifade bulunsa da ölçüm katmanı bunu destekleyen ya da çürüten bir işaret saptamadığından bu yalnızca söze dayalı bir izlenimdir."
               if emo else "Söylenen içeriğin duygu sınıflandırması bir ipucu vermedi.")
    else:
        soz = (f"Söylenen içerik ağırlıkla {SOZ_DUYGU_TR[emo.most_common(1)[0][0]]} duygusu taşıyor; ancak ölçüm katmanı bunu destekleyen ya da çürüten bir işaret saptamadığından bu yalnızca söze dayalı bir izlenimdir."
               if emo else "Söylenen içeriğin duygu sınıflandırması bir ipucu vermedi.")
    return (f"{parcalar[0]['t']} sn dahil ölçüm katmanı hiçbir parçada eşiği geçen yüz ya da ses işareti saptamadı; bu nedenle ölçümlere dayalı bir duygu çıkarımı yapılamıyor. "
            f"{soz}{eksik} Sonuç bu kesitler için değerlendirilemedi.")


def _kelimeler(s):
    return [w for w in re.findall(r"[\wçğıöşüÇĞİÖŞÜ']+", s.lower()) if len(w) >= 4]


def _kucuk_ilk(x):
    return x if not x else ("i" if x[0] == "İ" else x[0].lower()) + x[1:]


def kanit_dizisi_metni(ts, olcum):
    """modeling_davranis.kanit_dizisi_metni ile BIREBIR ayni bicim (birim testle dogrulanir). olcum: parca basina [(ifade, guven)]."""
    giris = [{"t": t, "olcum": ifade, "guven": round(float(p), 2)} for t, ol in zip(ts, olcum) for ifade, p in ol]
    if not giris:
        giris = [{"t": ts[0], "olcum": YOK_IFADE}]
    return json.dumps(giris, ensure_ascii=False, separators=(",", ":"))


def olay_ham(SB, imza, parcalar, kanit_esik=1.5, olcumler=None, olasi=None, guven_kurali=False):
    """parcalar: [{klip_id, t, etiket, olas, z (ham 20), mod, transkript, icerik}] -> (zaman_cizelgesi, olay, kanit_parca[(parca_idx, ifade)])
    ; olay.kanitlar[i] = {t, olcum, guven}. olcumler: parca basina [(ifade, guven)] (MODELIN olcum basligindan, egitimde out-of-fold);
    verilmezse gercek z'den (|z|>esik, guven=1.0) uretilir. Kanitlar yalnizca bu listeden; LLM yazmaz, modelin forward'inda kopyalanir."""
    zc, kanitlar, kanit_parca, parca_sonuc = [], [], [], []
    uyumlar, guvenler, duygular = [], [], []
    # olay.duygu 'cikarim yapilamiyor' olacaksa (yalniz cikarim: guven_kurali) isaret-anlami cumleleri YALNIZ kanit katmaninda yazilir (kullanici karari 2026-10-13)
    cikarim_yok = False
    if guven_kurali:
        en0 = Counter(p["etiket"] for p in parcalar).most_common(1)[0][0]
        cikarim_yok = sum(float(p["olas"].get(en0, 0.0)) for p in parcalar) / len(parcalar) < float(SB.GUVEN_ESIK[0])
    for pi, p in enumerate(parcalar):
        if olcumler is not None:
            satir = [(SB.TERS_IFADE[ifade][0], SB.TERS_IFADE[ifade][1], float(pg)) for ifade, pg in olcumler[pi]]
        else:
            satir = SB.satir_cikar(p["z"], p["mod"], esik=kanit_esik)
        yok_ = AK.yokluk_isaretleri()
        satir = [s_ for s_ in satir if SB.ifade_al(s_[0], s_[1]).lower() not in yok_]          # hareketin YOKLUGUNU anlatan isaretler kanit/olasi sayilmaz
        guven, _ = SB.son_guven(satir, p["etiket"], imza, float(max(p["olas"].values())), p["mod"])
        cev = SB.hedef_uret(p["klip_id"], p["etiket"], p["olas"], imza, satir, p["mod"], guven)
        kn_adlari = {SB.ifade_al(g, yn) for g, yn, _ in satir}
        ol_pi = [ifade for ifade, _ in (olasi[pi] if olasi is not None else []) if ifade not in kn_adlari]          # KANITTA olan isaret olasi listesinde OLAMAZ
        soz_ = p["transkript"] if p["mod"] != "sadece_goz" or p["transkript"] else ""
        kn_pi = [SB.ifade_al(g, yn) for g, yn, _ in satir[:6]]
        gmap = {SB.ifade_al(g, yn): (round(float(mag), 2) if olcumler is not None else 1.0) for g, yn, mag in satir[:6]}
        gmap.update({ifade: round(float(pg), 2) for ifade, pg in (olasi[pi] if olasi is not None else []) if ifade not in kn_adlari})
        # UYUM / birincil-ikincil anlam / yorum: KODLA (anlam_kurallari); klibin oyunculuk etiketi uyum icin KULLANILMAZ
        pd_ = AK.parca_degerlendir(p["mod"], p["t"], soz_, p.get("soz_duygu"), kn_pi, ol_pi, p.get("yokluk", []), gmap, icerik=p.get("icerik_duygusu"), yalniz_kanit_anlam=cikarim_yok)
        isaretsiz = not pd_["cozum"]                             # kanit da olasi da yok -> uyum DEGERLENDIRILEMEDI, yorum KODLA
        uyum = pd_["uyum"]
        yorum_s = isaretsiz_yorum(p) if isaretsiz else pd_["yorum"]
        dav = AK.davranis_metni(p["mod"], []) if isaretsiz else pd_["davranis"]
        parca_sonuc.append(pd_)
        zc.append({"t": p["t"], "soylenen": soz_, "davranis": dav, "yorum": yorum_s, "uyum": uyum})
        uyumlar.append(uyum); guvenler.append(guven); duygular.append(p["etiket"])
        for g, yn, mag in satir[:6]:
            gv = float(mag) if olcumler is not None else 1.0
            kanitlar.append({"t": p["t"], "olcum": SB.ifade_al(g, yn), "guven": round(gv, 2)})
            kanit_parca.append((pi, SB.ifade_al(g, yn)))
    if not kanit_parca:                       # hic olculmus isaret yok: sabit "olcum yok" kaydi (kanit sayilmaz)
        kanitlar.append({"t": parcalar[0]["t"], "olcum": "ölçülmüş belirgin işaret yok"})
        kanit_parca.append((0, "ölçülmüş belirgin işaret yok"))
    en_cok = Counter(duygular).most_common(1)[0][0]
    modlar = {p["mod"] for p in parcalar}
    girdi = ("girdi_yok" if not modlar else "goruntu+ses" if modlar == {"tam"} else
             "sadece_goruntu" if modlar == {"sadece_goz"} else "sadece_ses" if modlar == {"sadece_kulak"} else "goruntu+ses")
    if len(modlar) > 1 and "tam" not in modlar:
        girdi = "goruntu+ses"
    degerli = [u for u in uyumlar if u != "degerlendirilemedi"]
    uyum_o = AK.olay_uyumu([(r_["uyum"], r_["skor"]) for r_ in parca_sonuc])
    durum, skor = uyum_o["durum"], uyum_o["skor"]
    _, tutum_o = AK.ruh_tutum([r_["bir"] for r_ in parca_sonuc])
    # tekrar eden kaliplar: ayni OLCULMUS isaretin >=2 parcada gorulmesi (kural; LLM uretmez; model forward'i ile ayni islev)
    ol_parca = [[ifade for pj, ifade in kanit_parca if pj == pi and ifade != YOK_IFADE] for pi in range(len(parcalar))]
    kaliplar = KURAL.tekrar_kaliplari(ol_parca, uyumlar, [bool((p["transkript"] or "").strip()) for p in parcalar])
    olasi_liste = [{"t": parcalar[pi]["t"], "olcum": ifade, "guven": round(float(pg), 2)}          # OLASI katman: kodla doldurulur
                   for pi in range(len(parcalar)) for ifade, pg in (olasi[pi] if olasi is not None else [])
                   if (parcalar[pi]["t"], ifade) not in {(k_["t"], k_["olcum"]) for k_ in kanitlar}]
    # genel_yorum (kod yedegi) = ONEK (olcum cumleleri KESIN + guclu uyum varsa uyum cumlesi; sablonda ZORLANIR) + yorum cumleleri (temkinli; anlam_kurallari.kod_genel)
    onek_ = AK.genel_onek(kanitlar, olasi_liste, durum, ts=[p["t"] for p in parcalar], yok_ifade=YOK_IFADE)
    imli_ = [(p["t"], r_) for p, r_ in zip(parcalar, parca_sonuc) if r_["cozum"]]
    gy = AK.kod_genel(onek_, imli_, yalniz_kanit_anlam=cikarim_yok)
    hepsi_isaretsiz = not any(r_["cozum"] for r_ in parca_sonuc)      # hicbir parcada kanit/olasi isaret yok (ses-yok parcalar isaretli sayilir)
    duygu_o = SB.DUYGU_KOD[en_cok]                               # olay.duygu: klibin GERCEK etiketinden (egitim hedefi) / cikarimda modelin sinif basligindan (zorlanir); soze DAYANMAZ
    en_olasi = None
    if guven_kurali:                                             # YALNIZ cikarimda (egitim hedefi/ogretmen yolu degismez): en olasi sinifin olasiligi (parca ort.) < GUVEN_ESIK[0] -> duygu yazilmaz
        q = sum(float(p["olas"].get(en_cok, 0.0)) for p in parcalar) / len(parcalar)
        esik_g = float(SB.GUVEN_ESIK[0])
        en_olasi = {"sinif": en_cok, "p": round(q, 3), "esik": esik_g}
        if q < esik_g:
            duygu_o = CIKARIM_YOK
    ruh_o = AK.ruh_hali_kosullu([c_ for r_ in parca_sonuc for c_ in r_["cozum"]], duygu_o, duygu_yazildi=(duygu_o != CIKARIM_YOK))   # ruh_hali: duygu esigi gecti VE kanit katmani isareti tutarli; yoksa 'degerlendirilemedi'
    if hepsi_isaretsiz:                                          # isaret yok: genel_yorum kodla, guven 'dusuk'
        gy = isaretsiz_genel(parcalar, duygu_adi_yok=(duygu_o == CIKARIM_YOK))
    olay = {"girdi": girdi, "ruh_hali": ruh_o, "duygu": duygu_o,
            "guven": "düşük" if hepsi_isaretsiz else min(guvenler, key=SB.GUVEN_SIRA.index),
            "soz_davranis_uyumu": {"durum": durum, "skor": skor}, "tutum": tutum_o,
            "tekrar_eden_kaliplar": kaliplar, "kisilik_izlenimi": None, "kanitlar": kanitlar,
            "olasi_isaret": olasi_liste,
            "genel_yorum": gy}
    if en_olasi is not None:
        olay["en_olasi"] = en_olasi                              # TEKNIK alan (zorlanan sablon metnine girmez; sb_olay_kod secili anahtarlari yazar)
        if duygu_o == CIKARIM_YOK:
            olay["guven"] = "düşük"
    return zc, olay, kanit_parca


def olay_hedefi(SB, imza, parcalar, kanit_esik=1.5, olcumler=None, olasi=None):
    """-> (isaretli_metin, parca_kanit_idx). 'kanitlar' dizisi ISARET_ZA..ISARET_ZB ile sarilir (modeling_davranis.hedef_parcala: 'zorla'
    parcasi; egitimde kayip YOK, uretimde modelin olcum basligindan zorlanir)."""
    zc, olay, kanit_parca = olay_ham(SB, imza, parcalar, kanit_esik, olcumler, olasi)
    ts = [p["t"] for p in parcalar]
    ol = [[] for _ in parcalar]
    for pi, ifade in kanit_parca:
        if ifade != YOK_IFADE:
            gv = next(k["guven"] for k in olay["kanitlar"] if k["olcum"] == ifade and k["t"] == ts[pi])
            ol[pi].append((ifade, gv))
    dizi = kanit_dizisi_metni(ts, ol)
    olay["kanitlar"] = "@@KD@@"
    olasi_metni = json.dumps(olay["olasi_isaret"], ensure_ascii=False, separators=(",", ":"))      # modeling_davranis.olasi_dizisi_metni ile AYNI bicim
    olay["olasi_isaret"] = "@@OI@@"
    kalip_metni = KURAL.kalip_dizisi_metni(olay["tekrar_eden_kaliplar"])
    olay["tekrar_eden_kaliplar"] = "@@KP@@"
    metin = json.dumps({"zaman_cizelgesi": zc, "olay": olay}, ensure_ascii=False, separators=(",", ":"))
    metin = metin.replace('"@@KD@@"', ISARET_ZA + dizi + ISARET_ZB).replace('"@@KP@@"', ISARET_ZA + kalip_metni + ISARET_ZB).replace('"@@OI@@"', ISARET_ZA + olasi_metni + ISARET_ZB)
    return metin, [pi for pi, _ in kanit_parca]


HEDEF_PARCA_ALANLARI = ("t", "soylenen", "davranis", "yorum", "uyum")


def hedef_metni_json(j):
    """Ogretmen/sablon/kod_genel CIKTISI ({zaman_cizelgesi, olay}) -> EGITIM HEDEF metni (olay_hedefi ile ayni bicim: kanitlar / olasi_isaret / tekrar_eden_kaliplar
    ISARET_ZA..ISARET_ZB 'zorla' bolgeleri). 'cozumleme' ve kaynak alanlari egitim hedefine GIRMEZ (yalniz denetim icin). Dondurur (isaretli_metin, olcum, olasi): olcum/olasi =
    parca basina [(ifade, guven)] (egitimde modelin 'Olcumler:' satirlari AYNEN bunlardan kurulur -> hedefle tutarli)."""
    zc = [{k: e[k] for k in HEDEF_PARCA_ALANLARI} for e in j["zaman_cizelgesi"]]
    ts = [e["t"] for e in zc]
    olay = dict(j["olay"])
    olcum = [[(k["olcum"], float(k.get("guven", 1.0))) for k in olay["kanitlar"] if k["t"] == t and k["olcum"] != YOK_IFADE] for t in ts]
    olasi = [[(k["olcum"], float(k.get("guven", 0.0))) for k in olay.get("olasi_isaret", []) if k["t"] == t] for t in ts]
    dizi = kanit_dizisi_metni(ts, olcum)
    olasi_metni = json.dumps(olay.get("olasi_isaret", []), ensure_ascii=False, separators=(",", ":"))          # modeling_davranis.olasi_dizisi_metni ile AYNI bicim
    kalip_metni = KURAL.kalip_dizisi_metni(olay["tekrar_eden_kaliplar"])
    olay["kanitlar"] = "@@KD@@"
    olay["olasi_isaret"] = "@@OI@@"
    olay["tekrar_eden_kaliplar"] = "@@KP@@"
    metin = json.dumps({"zaman_cizelgesi": zc, "olay": olay}, ensure_ascii=False, separators=(",", ":"))
    metin = metin.replace('"@@KD@@"', ISARET_ZA + dizi + ISARET_ZB).replace('"@@KP@@"', ISARET_ZA + kalip_metni + ISARET_ZB).replace('"@@OI@@"', ISARET_ZA + olasi_metni + ISARET_ZB)
    return metin, olcum, olasi


# ------------------------------------------------------------------------------------------------ SABLON (kod-zorlanan JSON): LLM yalniz icerik_duygusu + genel_yorum uretir
ICERIK_DEGERLERI = ("ofke", "korku", "uzuntu", "tiksinme", "mutluluk", "sakin", "saskinlik", "notr", "yok")
ICERIK_ANAHTAR = {"ofke": "Ofke", "korku": "Korku", "uzuntu": "Uzuntu", "tiksinme": "Tiksinme", "mutluluk": "Mutluluk", "sakin": "Sakin", "saskinlik": "Saskinlik", "notr": None, "yok": None}


def _j(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def sb_parca_bas(i, t, soz):
    """Parca i'nin sabit baslangici: ... "icerik_duygusu":"  (LLM bundan sonrasini yazar: deger + kapanis tirnagi)."""
    return ('{"zaman_cizelgesi":[' if i == 0 else ",") + '{"t":' + _j(t) + ',"soylenen":' + _j(soz) + ',"icerik_duygusu":"'


def sb_parca_kod(e):
    """KOD: davranis, yorum, uyum (+ parca kapanisi)."""
    return ',"davranis":' + _j(e["davranis"]) + ',"yorum":' + _j(e["yorum"]) + ',"uyum":' + _j(e["uyum"]) + "}"


def sb_olay_kod(olay, ts):
    """KOD: olay alanlari ('genel_yorum' acilisina kadar). kanitlar/olasi_isaret/tekrar_eden_kaliplar eskisiyle AYNI bicim."""
    ol = [[(k["olcum"], float(k.get("guven", 1.0))) for k in olay["kanitlar"] if k["t"] == t and k["olcum"] != YOK_IFADE] for t in ts]
    return ('],"olay":{"girdi":' + _j(olay["girdi"]) + ',"ruh_hali":' + _j(olay["ruh_hali"]) + ',"duygu":' + _j(olay["duygu"]) + ',"guven":' + _j(olay["guven"])
            + ',"soz_davranis_uyumu":' + _j(olay["soz_davranis_uyumu"]) + ',"tutum":' + _j(olay["tutum"]) + ',"tekrar_eden_kaliplar":' + KURAL.kalip_dizisi_metni(olay["tekrar_eden_kaliplar"])
            + ',"kisilik_izlenimi":null,"kanitlar":' + kanit_dizisi_metni(ts, ol) + ',"olasi_isaret":' + _j(olay.get("olasi_isaret", [])) + ',"genel_yorum":"' + genel_onek_metni(olay, ts))


def genel_onek_metni(olay, ts):
    """genel_yorum'un KOD tarafindan yazilan on eki (JSON-kacisli, sonda bosluk): olcum cumleleri KESIN + guclu uyum cumlesi. Isaret yoksa ''. LLM bundan sonrasini (yorum + alternatif) yazar."""
    if hepsi_isaretsiz_mi(olay):
        return ""
    o = AK.genel_onek(olay["kanitlar"], olay.get("olasi_isaret", []), olay["soz_davranis_uyumu"]["durum"], ts=ts, yok_ifade=YOK_IFADE)
    return (_j(o + " ")[1:-1]) if o else ""


def hepsi_isaretsiz_mi(olay):
    return not olay.get("olasi_isaret") and all(k["olcum"] == YOK_IFADE for k in olay["kanitlar"])


def genel_govde(genel):
    return _j(genel)[1:-1] + '"}}'


def hedef_sablonu(j):
    """Cevap anahtari kaydi ({zaman_cizelgesi (+icerik_duygusu), olay}) -> EGITIM segmentleri [("zorla"|"m", metin)]: 'zorla' = KOD (kayip maskeli), 'm' = LLM'in ogrenecegi
    (icerik_duygusu degeri + kapanis tirnagi; genel_yorum + '"}}'). Cikarimdaki SablonUretici ile BIREBIR ayni metinleri uretir. Dondurur (segmentler, olcum, olasi)."""
    zc, olay = j["zaman_cizelgesi"], j["olay"]
    ts = [e["t"] for e in zc]
    olcum = [[(k["olcum"], float(k.get("guven", 1.0))) for k in olay["kanitlar"] if k["t"] == t and k["olcum"] != YOK_IFADE] for t in ts]
    olasi = [[(k["olcum"], float(k.get("guven", 0.0))) for k in olay.get("olasi_isaret", []) if k["t"] == t] for t in ts]
    seg = [("zorla", sb_parca_bas(0, zc[0]["t"], zc[0]["soylenen"]))]
    for i, e in enumerate(zc):
        seg.append(("m", e["icerik_duygusu"] + '"'))
        kod = sb_parca_kod(e)
        kod += sb_parca_bas(i + 1, zc[i + 1]["t"], zc[i + 1]["soylenen"]) if i + 1 < len(zc) else sb_olay_kod(olay, ts)
        if i + 1 == len(zc) and hepsi_isaretsiz_mi(olay):
            kod += genel_govde(olay["genel_yorum"])                       # isaret yok: genel_yorum da KODLA
        seg.append(("zorla", kod))
    if not hepsi_isaretsiz_mi(olay):
        onek = AK.genel_onek(olay["kanitlar"], olay.get("olasi_isaret", []), olay["soz_davranis_uyumu"]["durum"], ts=ts, yok_ifade=YOK_IFADE)
        g = olay["genel_yorum"]
        if onek:
            assert g.startswith(onek), f"genel_yorum kod onekiyle baslamali: {onek!r} / {g[:80]!r}"
            g = g[len(onek):].lstrip()
        seg.append(("m", genel_govde(g)))
    return seg, olcum, olasi


class SablonUretici:
    """CIKARIM: modelin olcumleri + sinif basligi + (LLM'in urettigi) icerik_duygusu degerlerinden KOD alanlarini hesaplar ve zorlanacak metinleri verir (olay_ham ile ayni islev)."""

    def __init__(self, SB, imza, parcalar, olcum, olasi):
        self.SB, self.imza, self.parcalar, self.olcum, self.olasi = SB, imza, [dict(p) for p in parcalar], olcum, olasi
        self.n = len(parcalar)
        self.ts = [p["t"] for p in parcalar]
        self.degerler = []
        self.bitti = False                                           # True: genel_yorum da kodla yazildi (LLM yalniz bitis token'i uretir)
        self.guven_kurali = True                                     # cikarim: olay.duygu GUVEN_ESIK[0] altinda yazilmaz (olay_ham guven_kurali)
        self.zc = []

    @classmethod
    def olustur(cls, SB, imza, ts, modlar, transkriptler, sinif_p, siniflar, olcum, olasi, klip_idleri=None, etiketler=None):
        """sinif_p: [C,len(siniflar)] olasiliklar (parca basina sinif basligi)."""
        parcalar = []
        for i in range(len(ts)):
            pr = [float(x) for x in sinif_p[i]]
            parcalar.append({"klip_id": (klip_idleri[i] if klip_idleri else f"cikarim_{i}"), "t": ts[i], "etiket": (etiketler[i] if etiketler else siniflar[max(range(len(pr)), key=lambda a: pr[a])]),
                             "olas": {s: p for s, p in zip(siniflar, pr)}, "mod": modlar[i], "transkript": transkriptler[i], "yokluk": []})
        return cls(SB, imza, parcalar, olcum, olasi)

    def _soz(self, p):
        return p["transkript"] if p["mod"] != "sadece_goz" or p["transkript"] else ""

    def ilk(self):
        p = self.parcalar[0]
        return sb_parca_bas(0, p["t"], self._soz(p))

    def parca_devam(self, i, deger):
        deger = deger if deger in ICERIK_DEGERLERI else "notr"
        p = self.parcalar[i]
        p["soz_duygu"], p["icerik_duygusu"] = ICERIK_ANAHTAR[deger], deger
        self.degerler.append(deger)
        zc, _, _ = olay_ham(self.SB, self.imza, [p], olcumler=[self.olcum[i]], olasi=[self.olasi[i]])
        self.zc.append(zc[0])
        metin = sb_parca_kod(zc[0])
        if i + 1 < self.n:
            q = self.parcalar[i + 1]
            return metin + sb_parca_bas(i + 1, q["t"], self._soz(q)), False
        _, olay, _ = olay_ham(self.SB, self.imza, self.parcalar, olcumler=self.olcum, olasi=self.olasi, guven_kurali=self.guven_kurali)
        self.olay = olay
        metin += sb_olay_kod(olay, self.ts)
        if hepsi_isaretsiz_mi(olay):
            metin += genel_govde(olay["genel_yorum"])
            self.bitti = True
        return metin, True

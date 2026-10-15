r"""Isaret ANLAM COZUMLEME + UYUM + BIRINCIL/IKINCIL ANLAM + ruh_hali/tutum + parca yorumu: TEK kaynak (v2/veri/anlam_cozumleme.json).
hedef_v2.olay_ham (egitim hedefi / sablon yedegi), ogretmen_v2 (ogretmen hatti) ve model (modeling_davranis) AYNI islevleri kullanir.
Klibin oyunculuk etiketi uyum icin KULLANILMAZ: uyum, GOZLENEN davranisin birincil anlami ile soylenen icerigin ailesinden hesaplanir.
Bagimlilik yok (yalniz standart kutuphane)."""
import json
import os
import re

_C = None
PRI = ["ofke", "tiksinme", "uzuntu", "korku", "olumlu", "saskinlik", "olumsuz_genel", "bilissel"]      # birincil_anlam: aile onceligi
ICERIK_TR = {"ofke": "öfke", "korku": "korku", "uzuntu": "üzüntü", "tiksinme": "tiksinme", "mutluluk": "mutluluk", "sakin": "sakinlik", "saskinlik": "şaşkınlık"}
ON = {"sadece_goz": "ses girdisi yok; ", "sadece_kulak": "görüntü girdisi yok; "}
RUH_DEGER = {"ofke": ["öfkeli", "gergin"], "tiksinme": ["tiksinmiş", "rahatsız"], "uzuntu": ["üzgün", "keyifsiz"], "korku": ["kaygılı", "gergin"],
             "olumlu": ["neşeli"], "saskinlik": ["şaşkın"], "olumsuz_genel": ["rahatsız"], "bilissel": ["kararsız"]}
TUTUM_AILE = {"ofke": "baskin", "tiksinme": "geri_cekilmis", "uzuntu": "geri_cekilmis", "korku": "savunmaci", "olumlu": "rahat",
              "saskinlik": "tarafsiz", "olumsuz_genel": "kacingan", "bilissel": "tarafsiz"}


def yol_bul():
    ust = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (ust, os.path.join(ust, "veri"), os.environ.get("DAVRANIS_KOK", "")):
        if d and os.path.isdir(d) and os.path.exists(os.path.join(d, "anlam_cozumleme.json")):
            return os.path.join(d, "anlam_cozumleme.json")
    raise FileNotFoundError("anlam_cozumleme.json bulunamadi")


def kurallar():
    global _C
    if _C is None:
        with open(yol_bul(), encoding="utf-8") as f:
            _C = json.load(f)
    return _C


def yokluk_isaretleri():
    return {x.lower() for x in kurallar()["yokluk_isaretleri"]["liste"]}


def aile(anlam):
    return kurallar()["anlam_ailesi"].get(anlam, "bilissel")


def degerlik(a):
    return kurallar()["aile_degerlik"].get(a, "notr")


def icerik_etiketi(mod, transkript, soz_duygu):
    """Soylenen cumlenin etiketi (ofke, korku, uzuntu, tiksinme, mutluluk, sakin, saskinlik, notr); soylenen yoksa / ses yoksa 'yok'. soz_duygu: banka duygu anahtari ('Ofke'...) ya da None."""
    soz = (transkript or "").strip()
    if mod == "sadece_goz" or not soz or soz == "-":
        return "yok"
    return soz_duygu.lower() if soz_duygu else "notr"


def cozumle(ifade, bagl, icerik):
    """ifade icin ilk eslesen kural -> {anlam, alternatif, kural}. bagl: ayni parcadaki (kanit+olasi+yokluk) isaret kumesi (kucuk harf)."""
    for k, r in enumerate(kurallar()["isaretler"].get(ifade, [])):
        if "varsa_herhangi" in r and not (set(r["varsa_herhangi"]) & (bagl - {ifade})):
            continue
        if "yoksa_hepsi" in r and (set(r["yoksa_hepsi"]) & bagl):
            continue
        if "icerik" in r and icerik not in r["icerik"]:
            continue
        return {"anlam": r["anlam"], "alternatif": r["alternatif"], "kural": k}
    return None


def parca_cozumleri(kanit, olasi, yokluk, icerik, guven=None):
    """[{isaret, anlam, alternatif, katman: kanıt|olası, kural, aile, guven}]; yokluk isaretleri cozumlenmez (yalniz baglam). guven: {ifade: P}."""
    yok = yokluk_isaretleri()
    kanit = list(dict.fromkeys(x.lower() for x in kanit if x.lower() not in yok))                          # tekrarlar atilir
    olasi = list(dict.fromkeys(x.lower() for x in olasi if x.lower() not in yok and x.lower() not in kanit))      # KANITTA olan isaret olasi listesinde OLAMAZ
    bagl = set(kanit) | set(olasi) | {x.lower() for x in (yokluk or [])}
    g = {k.lower(): v for k, v in (guven or {}).items()}
    out = []
    for katman, lst in (("kanıt", kanit), ("olası", olasi)):
        for f in lst:
            c = cozumle(f, bagl, icerik)
            if c:
                out.append({"isaret": f, **c, "katman": katman, "aile": aile(c["anlam"]), "guven": float(g.get(f, 0.0))})
    return out


def birincil_ikincil(coz):
    """birincil_anlam tanimi: once kanit katmani, sonra olasi; ayni katmanda aile onceligi, esitlikte guveni yuksek olan. Ikincil: birincilden FARKLI ailedeki en oncelikli anlam."""
    if not coz:
        return None, None
    ana = [c for c in coz if c["katman"] == "kanıt"] or list(coz)
    bir = min(ana, key=lambda c: (PRI.index(c["aile"]), -c["guven"]))
    diger = [c for c in coz if c["aile"] != bir["aile"] and c["isaret"] != bir["isaret"]]            # ikincil: birincilin geldigi isaretten FARKLI bir isaret (yoksa ikincil yok)
    ik = min(diger, key=lambda c: (PRI.index(c["aile"]), c["katman"] != "kanıt", -c["guven"])) if diger else None
    return bir, ik


UYUM_ACIK = False        # v2.1 KARARI (kullanici 2026-10-14): MUStARD++ dev AUC 0,50 -> uyum iddiasi KAPALI; v2.1 dev'de AUC >= 0,60 olursa True yapilir.


def uyum_parca(coz, bir, icerik):
    """-> (uyum, skor). v2.1 UYUM KARARI (v21_plan 7.7; kullanici 2026-10-13): YALNIZ birincil anlam KANIT katmanindaysa ve icerik duygusu belirliyse karar verilir:
    uyumlu = ayni aile (belirli aile; olumsuz_genel/bilissel jokeri ile uyumlu DENMEZ), uyumsuz = zit degerlik (olumlu <-> olumsuz; saskinlik/bilissel haric);
    diger her durumda 'degerlendirilemedi' ('kismen' uretilmez). Klibin oyunculuk etiketi KULLANILMAZ."""
    if not UYUM_ACIK or not coz or bir is None or icerik in ("yok", "notr") or bir["katman"] != "kanıt":
        return "degerlendirilemedi", None
    fam = bir["aile"]
    ia = kurallar()["icerik_ailesi"].get(icerik, "notr")
    if fam in ("bilissel", "saskinlik") or ia in ("notr", "bilissel", "saskinlik"):
        return "degerlendirilemedi", None
    if fam == ia and fam != "olumsuz_genel":
        return "uyumlu", 1.0
    if {degerlik(fam), degerlik(ia)} == {"olumlu", "olumsuz"}:
        return "uyumsuz", 0.0
    return "degerlendirilemedi", None


def olay_uyumu(parca_uyumlari):
    """parca_uyumlari: [(uyum, skor)] -> {durum, skor}: isaretli parcalarin ortalamasi; hepsi degerlendirilemedi ise degerlendirilemedi. durum: >=0.7 uyumlu, <=0.3 uyumsuz, aksi kismen."""
    sk = [s for u, s in parca_uyumlari if u != "degerlendirilemedi" and s is not None]
    if not sk:
        return {"durum": "degerlendirilemedi", "skor": None}
    m = round(sum(sk) / len(sk), 2)
    return {"durum": "uyumlu" if m >= 0.7 else ("uyumsuz" if m <= 0.3 else "kismen"), "skor": m}


def ruh_tutum(birinciller):
    """birinciller: isaretli parcalarin birincil anlam kayitlari. ruh_hali_kurali: en sik iki aile -> degerler; tutum: en sik ailenin tutumu. -> (ruh_hali metni, tutum). Isaret yoksa (degerlendirilemedi, tarafsiz)."""
    birinciller = [b for b in birinciller if b]
    if not birinciller:
        return "değerlendirilemedi", "tarafsiz"
    sayac = {}
    for b in birinciller:
        sayac[b["aile"]] = sayac.get(b["aile"], 0) + 1
    sirali = sorted(sayac, key=lambda a: (-sayac[a], PRI.index(a)))
    yalniz_olasi = all(b["katman"] == "olası" for b in birinciller)

    def deger(a):
        if a == "olumlu":
            m = " ".join(b["anlam"].lower() for b in birinciller if b["aile"] == "olumlu")
            return ["heyecanlı"] if "heyecan" in m else (["rahat"] if re.search(r"rahatlık|sakinlik", m) else ["neşeli"])
        return RUH_DEGER[a]
    if len(sirali) == 1:
        vals = deger(sirali[0])
    else:
        vals = [deger(sirali[0])[0], deger(sirali[1])[0]]
    vals = list(dict.fromkeys(vals))[:2]
    if yalniz_olasi:
        vals = vals[:1]
    tut = TUTUM_AILE[sirali[0]]
    if sirali[0] == "olumlu" and any("heyecan" in b["anlam"].lower() for b in birinciller if b["aile"] == "olumlu"):
        tut = "kendinden_emin"
    return ", ".join(vals), tut


DUYGU_AILE = {"ofkeli": "ofke", "korkmus": "korku", "uzgun": "uzuntu", "mutlu": "olumlu", "tiksinmis": "tiksinme", "saskin": "saskinlik"}      # olay.duygu kodu -> anlam ailesi (sakin/notr/belirsiz: ailesi yok)
KANIT_KATMANI_ETIKETI = "(kanıt katmanı)"


def ruh_hali_kosullu(cozumler, duygu_kod, duygu_yazildi=True):
    """olay.ruh_hali (kullanici karari 2026-10-13): YALNIZ olay.duygu guven esigini gecmisse (duygu_yazildi) VE en az bir KANIT katmani isareti o duyguyla tutarliysa
    (isaretin anlam ailesi == duygunun ailesi) yazilir; katman belirtilir. Aksi halde 'degerlendirilemedi'. cozumler: parca cozumleri (parca_degerlendir()['cozum']) birlesimi."""
    if not duygu_yazildi:
        return "değerlendirilemedi"
    fam = DUYGU_AILE.get(duygu_kod)
    kanit = [c for c in cozumler if c.get("katman") == "kanıt"]
    if fam is None or not any(c["aile"] == fam for c in kanit):
        return "değerlendirilemedi"
    uyan = [c for c in kanit if c["aile"] == fam]
    if fam == "olumlu":
        m = " ".join(c["anlam"].lower() for c in uyan)
        deger = "heyecanlı" if "heyecan" in m else ("rahat" if re.search(r"rahatlık|sakinlik", m) else "neşeli")
    else:
        deger = RUH_DEGER[fam][0]
    return f"{deger} {KANIT_KATMANI_ETIKETI}"


def de_da(s):
    """Son unluye gore ayri yazilan 'de'/'da' baglaci: 'gözler iri açık' -> 'da', 'öfke' -> 'de'."""
    for ch in reversed((s or "").strip().lower()):
        if ch in "aıouâû":
            return "da"
        if ch in "eiöüî":
            return "de"
    return "de"


def olcum_ifadesi(kanit, olasi):
    """OLCUM KESIN (temkin yalniz yorumda; kullanici 2026-10-13): 'ölçümde A, B belirgin (kanıt); C hafif (olası)'. 'olabilir' icermez."""
    parca = []
    if kanit:
        parca.append(", ".join(kanit) + " belirgin (kanıt)")
    if olasi:
        parca.append(", ".join(olasi) + " hafif (olası)")
    return "ölçümde " + "; ".join(parca)


def ilk_cumle(mod, uyum, zaman, soylenen, isaretler, olasi=()):
    """Isaretli parca yorumunun 1. cumlesi (olcum, KESIN): 'ölçümde X belirgin (kanıt)' / 'ölçümde Y hafif (olası)'. '"..." diyor ancak' YALNIZ uyum=uyumsuz parcada."""
    on = ON.get(mod, "")
    var_soz = bool((soylenen or "").strip()) and (soylenen or "").strip() != "-" and mod != "sadece_goz"
    o = olcum_ifadesi(isaretler, [x for x in olasi if x not in isaretler])
    return (f"{zaman} sn: {on}\"{soylenen}\" diyor ancak {o}." if uyum == "uyumsuz" and var_soz else f"{zaman} sn: {on}{o}.")


def parca_yorumu(mod, zaman, soylenen, kanit, olasi, coz, bir, ik, uyum, icerik, yalniz_kanit_anlam=False):
    """ISARETLI parcanin yorumu TAMAMEN KODLA: gozlem + cikarim (+ ikincil) (+ icerik bagi) + alternatif (kaliplar kullanici tanimli)."""
    kanit = [x for x in kanit if x.lower() not in yokluk_isaretleri()]
    olasi = [x for x in olasi if x.lower() not in yokluk_isaretleri() and x not in kanit]
    s = [ilk_cumle(mod, uyum, zaman, soylenen, kanit, olasi)]
    # ISARET-ANLAMI cumleleri KISIYE degil ISARETE atif yapar (kullanici karari 2026-10-13): duygu adi kisinin durumu gibi okunmaz.
    # yalniz_kanit_anlam=True (olay.duygu 'cikarim yapilamiyor'): anlam cumleleri YALNIZ kanit katmaninda yazilir; olasi katmanda yalniz gozlem.
    if bir["katman"] == "kanıt" or not yalniz_kanit_anlam:
        on_ek = "Bu ölçüm" if bir["katman"] == "kanıt" else "Bu olası ölçüm"
        s.append(f"{on_ek} genelde {bir['anlam']} ile ilişkilendirilebilir; tek başına duygu kanıtı değildir.")
        if ik and (ik["katman"] == "kanıt" or not yalniz_kanit_anlam):
            nit = "ölçümü" if ik["katman"] == "kanıt" else "olası ölçümü"
            s.append(f"{ik['isaret'][:1].upper() + ik['isaret'][1:]} {nit} ise genelde {ik['anlam']} ile ilişkilendirilebilir.")
    ic = ICERIK_TR.get(icerik)
    if ic and uyum == "uyumlu":
        s.append(f"Söylenen içeriğin duygusu {ic}; bu, ölçülen davranışla aynı yönde.")
    elif ic and uyum == "uyumsuz":
        s.append(f"Söylenen içeriğin duygusu {ic}; bu, ölçülen davranışla çelişiyor.")
    if bir["katman"] == "kanıt" or not yalniz_kanit_anlam:
        s.append(f"Alternatif olarak {bir['alternatif']} {de_da(bir['alternatif'])} düşünülebilir.")
    return " ".join(s)


def davranis_metni(mod, kanit):
    kanit = [x for x in kanit if x.lower() not in yokluk_isaretleri()]
    return ON.get(mod, "") + (", ".join(kanit) if kanit else "eşiği geçen ölçüm yok")


def parca_degerlendir(mod, zaman, soylenen, soz_duygu, kanit, olasi, yokluk=(), guven=None, icerik=None, yalniz_kanit_anlam=False):
    """Bir parcanin TUM kodla uretilen sonucu: {cozum, bir, ik, uyum, skor, icerik, yorum, davranis}. Isaret yoksa yorum/davranis None (cagiran isaretsiz kalibi kullanir)."""
    ic = icerik if icerik else icerik_etiketi(mod, soylenen, soz_duygu)          # icerik: modelin urettigi icerik_duygusu (verilirse transkript/banka etiketi yerine)
    coz = parca_cozumleri(kanit, olasi, yokluk, ic, guven)
    bir, ik = birincil_ikincil(coz)
    uyum, skor = uyum_parca(coz, bir, ic)
    r = {"cozum": coz, "bir": bir, "ik": ik, "uyum": uyum, "skor": skor, "icerik": ic, "yorum": None, "davranis": None}
    if coz:
        izin = [c["isaret"] for c in coz if c["katman"] == "kanıt"]
        ol = [c["isaret"] for c in coz if c["katman"] == "olası"]
        r["yorum"] = parca_yorumu(mod, zaman, soylenen, izin, ol, coz, bir, ik, uyum, ic, yalniz_kanit_anlam)
        r["davranis"] = davranis_metni(mod, izin)
    return r


UYUM_CUMLESI = {"uyumlu": "Söylenen içeriğin duygusu ölçülen davranışla aynı yönde (kanıt düzeyinde).",
                "uyumsuz": "Söylenen içeriğin duygusu ölçülen davranışla çelişiyor (kanıt düzeyinde)."}


def uyum_cumlesi(durum):
    """Uyum cumlesi YALNIZ KOD tarafindan ve YALNIZ guclu (kanit duzeyi) uyumlu/uyumsuz kararda yazilir; aksi halde genel_yorumda uyumdan hic soz edilmez ('')."""
    return UYUM_CUMLESI.get(durum, "")


def genel_onek(kanitlar, olasi_isaret, uyum_durum, ts=None, yok_ifade="ölçülmüş belirgin işaret yok", en_cok=2):
    """genel_yorum'un KOD tarafindan yazilan (sablonda ZORLANAN) on eki: parca basina '<t> sn aralığında ölçümde ... belirgin (kanıt); ... hafif (olası).' (en fazla en_cok isaretli parca)
    + (guclu uyum varsa) uyum cumlesi. Yalniz olay alanlarindan (kanitlar, olasi_isaret, soz_davranis_uyumu) hesaplanir: egitim hedefi, ogretmen ve cikarim AYNI islev.
    Hic isaret yoksa ''."""
    if ts is None:
        ts = list(dict.fromkeys([k["t"] for k in kanitlar] + [k["t"] for k in olasi_isaret]))
    s = []
    for t_ in ts:
        kn = list(dict.fromkeys(k["olcum"] for k in kanitlar if k["t"] == t_ and k["olcum"] != yok_ifade))
        ol = [o["olcum"] for o in olasi_isaret if o["t"] == t_ and o["olcum"] not in kn]
        ol = list(dict.fromkeys(ol))
        if kn or ol:
            s.append(f"{t_} sn aralığında {olcum_ifadesi(kn, ol)}.")
    if not s:
        return ""
    s = s[:en_cok]
    u = uyum_cumlesi(uyum_durum)
    return " ".join(s + ([u] if u else []))


def kod_genel(onek, imli, yalniz_kanit_anlam=False):
    """DETERMINISTIK genel_yorum (LLM reddedilirse yedek; kaynak='kod_genel') = ONEK (genel_onek: olcum cumleleri [+ guclu uyum]) + 2 yorum cumlesi (LLM yoluyla AYNI kalip):
    'Bu [olası] ölçümler genelde <anlam> ile ilişkilendirilebilir.' + 'Alternatif olarak <alternatif> de düşünülebilir.'
    yalniz_kanit_anlam=True ve kanit yoksa yalniz onek (anlam cumlesi yok)."""
    if not imli:
        return onek
    if yalniz_kanit_anlam and not any(r["bir"]["katman"] == "kanıt" for _, r in imli):
        return onek
    sayac = {}
    for _, r in imli:
        b = r["bir"]
        if yalniz_kanit_anlam and b["katman"] != "kanıt":
            continue
        e = sayac.setdefault(b["anlam"], {"n": 0, "b": b, "kanit": False})
        e["n"] += 1
        e["kanit"] = e["kanit"] or b["katman"] == "kanıt"
    if not sayac:
        return onek
    en = min(sayac.values(), key=lambda e: (-e["n"], PRI.index(e["b"]["aile"]), -e["b"]["guven"]))
    anlam_c = (f"Bu ölçümler genelde {en['b']['anlam']} ile ilişkilendirilebilir." if en["kanit"]
               else f"Bu olası ölçümler genelde {en['b']['anlam']} ile ilişkilendirilebilir.")          # kalip LLM yoluyla AYNI (kullanici 2026-10-13); 'tek basina duygu kaniti degildir' parca yorumunda
    return " ".join(x for x in (onek, anlam_c, alternatif_cumlesi(en["b"])) if x)


def alternatif_cumlesi(b):
    """genel_yorum'un 'Alternatif olarak X de düşünülebilir.' cumlesi: KODLA, ana yorumun dayandigi BIRINCIL anlamin (bir) sozluk alternatifinden (kullanici 2026-10-14)."""
    return f"Alternatif olarak {b['alternatif']} {de_da(b['alternatif'])} düşünülebilir."


def birincil_secimi(imli, yorum_metni=None, yalniz_kanit_anlam=False):
    """imli: [(t, r)] -> ana yorumun dayandigi birincil anlam kaydi (bir). yorum_metni verilirse (LLM yorumu) metinde GECEN anlamlar arasindan en cok parcada gecen secilir;
    yoksa / eslesme yoksa kod_genel ile AYNI secim (en cok parca, aile onceligi, guven)."""
    sayac = {}
    for _, r in imli:
        b = r["bir"]
        if yalniz_kanit_anlam and b["katman"] != "kanıt":
            continue
        e = sayac.setdefault(b["anlam"], {"n": 0, "b": b})
        e["n"] += 1
    if not sayac:
        return None
    adaylar = list(sayac.values())
    if yorum_metni:
        ml = yorum_metni.lower()
        gecen = [e for e in adaylar if e["b"]["anlam"].lower() in ml]
        if gecen:
            adaylar = gecen
    return min(adaylar, key=lambda e: (-e["n"], PRI.index(e["b"]["aile"]), -e["b"]["guven"]))["b"]

"""v2.1.1 ISARET KATMANI (CIKARIMDA KOD): dogrudan_olcum.isaretler(...) ciktisi -> (kanit, olasi, yokluk) ifade listeleri.
Egitimdeki (J6b/J6c: ogretmen_v2.parca_olcumleri + olcum_kaydi) ile AYNI esleme ve modalite suzgeci; cikarimda ogrenci olcum basligi DEGIL bu liste
SablonUretici/girdi_kur'a 'olcum_override'/'olasi_override' olarak verilir. Olcum cumlesi sablonda zorlanir (kanit dizisi KOPYALANIR).
Kullanim (cikarim):  ol = dogrudan_olcum.isaretler(yuz_ham, ses_ham, pop, gecmis)  ->  K, O, Y = listeler(ol, mod, SB, yokluk_isaretleri)
    ornek['olcum_override'] = [K]; ornek['olasi_override'] = [O]   (parca basina bir liste; (ifade, guven=1.0) ciftleri)
NOT: gecmis.ekle(...) isaretler'den SONRA cagrilir (nedensel kisi referansi)."""


def ifade_kaydi(ol, SB):
    """isaretler -> (kanit [(ifade)], olasi [(ifade)]) modalite suzgeci OLMADAN (J6b.olcum_kaydi)."""
    f = lambda x: SB.IFADE[x["sinyal"]][0 if x["yon"] >= 0 else 1]
    kan = [f(x) for x in ol if x["katman"] == "kanit"]
    ola = [f(x) for x in ol if x["katman"] == "olasi"]
    return [x for x in kan if x], [x for x in ola if x]


def listeler(ol, mod, SB, yokluk):
    """-> (kanit [(ifade, 1.0)], olasi [(ifade, 1.0)], yokluk_baglami [ifade]). yokluk: {kucuk harf yokluk isaretleri}. Modalite: SB.mevcut_gruplar(mod)."""
    kan, ola = ifade_kaydi(ol, SB)
    var = SB.mevcut_gruplar(mod)
    ok = lambda x: SB.TERS_IFADE[x][0] in var
    k = [(x, 1.0) for x in kan if ok(x) and x.lower() not in yokluk]
    o = [(x, 1.0) for x in ola if ok(x) and x.lower() not in yokluk]
    y = [x for x in kan + ola if ok(x) and x.lower() in yokluk]
    return k, o, y

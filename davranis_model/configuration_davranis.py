"""DavranisModel yapilandirmasi: tek config.json altinda ViT (duygu7), Whisper, Qwen2.5 ve yardimci basliklar."""
from transformers import PretrainedConfig

SINIFLAR = ["Korku", "Mutluluk", "Notr", "Ofke", "Sakin", "Saskinlik", "Tiksinme", "Uzuntu"]

# yardimci baslik z-sirasi = yardimci_hedefler.csv sutun sirasi; her grup icin (yukselir, duser) ifadesi.
# Model bu tabloyu kanit kisitlamasi (kanit-ceza kaybi ve cikarimda mantik filtresi) icin config'ten okur.
IFADE = {
    "browDown": ("kaşlar çatık", "kaşlar çatılmıyor"),
    "browInnerUp": ("kaşların iç uçları yukarı çekilmiş", None),
    "browOuterUp": ("kaşlar yukarı kalkmış", None),
    "cheekSquint": ("yanaklar yukarı kalkmış", "yanaklarda hareket yok"),
    "eyeSquint": ("gözler kısık", "göz çevresi gevşek"),
    "eyeWide": ("gözler iri açık", "gözler normalden az açık"),
    "jawOpen": ("ağız belirgin şekilde açık", "ağız neredeyse hiç açılmıyor"),
    "mouthFrown": ("dudak köşeleri aşağı sarkmış", None),
    "mouthFunnel": ("dudaklar yuvarlak ve öne doğru büzülmüş", None),
    "mouthPucker": ("dudaklar büzük", None),
    "mouthPress": ("dudaklar birbirine bastırılmış", "dudaklar gevşek"),
    "mouthShrugLower": ("alt dudak ve çene yukarı itilmiş", None),
    "mouthSmile": ("belirgin bir gülümseme", "gülümsemeden eser yok"),
    "mouthStretch": ("dudaklar yanlara gerilmiş", None),
    "mouthUpperUp": ("üst dudak yukarı kalkmış", "üst dudak yerinde duruyor"),
    "noseSneer": ("burun kırışmış", "burun çevresi gevşek"),
    "rms_ortalama": ("ses belirgin şekilde yüksek", "ses kısık ve alçak"),
    "sessizlik_orani": ("konuşmada uzun duraksamalar", "neredeyse hiç duraksamadan konuşuyor"),
    "perde_ortalama_hz": ("ses tonu normalinden tiz", "ses tonu normalinden pes"),
    "perde_std_hz": ("ses tonu çok değişken", "ses tonu tekdüze"),
    # BAS/BAKIS (ogrenilmez; BlazeFace 6 noktadan geometrik, kisi bazli z): olcum basligi bunlari TAHMIN ETMEZ, forward'da hesaplanir
    "bas_yaw": ("başı yana dönük", None),
    "asagi_bakma": ("aşağı bakıyor", None),
    "bakis_kacirma": ("bakışını kaçırıyor", None),
    "bas_geri_cekilme": ("başını geri çekiyor", None),
}
BAS_GRUPLARI = ("bas_yaw", "asagi_bakma", "bakis_kacirma", "bas_geri_cekilme")      # olcum_kurallari.BAS_GRUPLARI ile ayni (birim testle dogrulanir)
SES_GRUPLARI = ("rms_ortalama", "sessizlik_orani", "perde_ortalama_hz", "perde_std_hz")

SISTEM = ("Sen bir kişinin görüntüsünü ve sesini algılayıp davranışını gerekçeli biçimde yorumlayan bir "
          "davranış analistisin. Yalnızca algıladığın işaretlere dayan; teşhis koyma, yalan tespiti yapma, "
          "kişilik için yalnızca 'izlenim' de.")


class DavranisConfig(PretrainedConfig):
    model_type = "davranis"

    def __init__(self, llm_config=None, vit_config=None, whisper_config=None, whisper_generation_config=None,
                 n_goz=8, n_kulak=16, n_kare=8, yuz_boyut=224, chunk_sn=3.0, ses_sr=16000,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, llm_adi="", llm_nf4=False,
                 siniflar=None, z_gruplari=None, isaret_ifadeleri=None, ses_gruplari=None,
                 kanit_esik=1.5, kisilik_aktif=False, kisilik_alanlari=None,
                 yuz_min_skor=0.5, yuz_pay=0.15, ses_sessiz_rms=1e-4, sistem=SISTEM,
                 kanit_p_min=0.5, olcum_max=6, kanit_sozlugu=None, bas_norm=None, olasi_p_min=0.7, olasi_max=3, guvenilmez_gruplar=("bas_geri_cekilme",), **kwargs):
        self.llm_config = llm_config or {}
        self.vit_config = vit_config or {}
        self.whisper_config = whisper_config or {}
        self.whisper_generation_config = whisper_generation_config or {}
        self.n_goz, self.n_kulak, self.n_kare = n_goz, n_kulak, n_kare
        self.yuz_boyut, self.chunk_sn, self.ses_sr = yuz_boyut, chunk_sn, ses_sr
        self.llm_adi, self.llm_nf4 = llm_adi, llm_nf4
        self.lora_r, self.lora_alpha, self.lora_dropout = lora_r, lora_alpha, lora_dropout
        self.siniflar = siniflar or list(SINIFLAR)
        self.z_gruplari = z_gruplari or list(IFADE)
        self.isaret_ifadeleri = isaret_ifadeleri or {g: list(v) for g, v in IFADE.items()}
        self.ses_gruplari = ses_gruplari or list(SES_GRUPLARI)
        self.kanit_esik = kanit_esik
        self.kisilik_aktif = kisilik_aktif      # ChaLearn lisans/onay bekleniyor: kapali; yapi (baslik) hazir
        self.kisilik_alanlari = kisilik_alanlari or ["disadonukluk", "uyumluluk", "sorumluluk", "duygusal_denge", "aciklik"]
        self.yuz_min_skor, self.yuz_pay, self.ses_sessiz_rms = yuz_min_skor, yuz_pay, ses_sessiz_rms
        self.sistem = sistem
        self.kanit_p_min, self.olcum_max = kanit_p_min, olcum_max      # kanit_p_min: float ya da {grup: [p_yukari, p_asagi]} (asama 0 val'de kanit dogrulugu >=0.85 icin secilir); olcum basligi: P>=p_min ve |z|>kanit_esik; en cok olcum_max isaret/parca
        self.kanit_sozlugu = kanit_sozlugu                            # None = tum isaretler; asama 0 val korelasyonu <0.3 olanlar CIKARILIR
        self.olasi_p_min, self.olasi_max = olasi_p_min, olasi_max            # OLASI katman: olasi_p_min <= P < kanit_p_min (kesin dille anilamaz)
        self.guvenilmez_gruplar = tuple(guvenilmez_gruplar)      # kanit olarak SUNULMAYAN gruplar (E0: bas_geri_cekilme MediaPipe ile r=0.15)
        self.bas_norm = bas_norm                                      # baş isaretleri kisi-norm tablosu {kisi|'genel': [[ort,sd]x4]} (None = genel varsayilan)
        super().__init__(**kwargs)

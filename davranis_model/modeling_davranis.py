"""DavranisModel: ham video karesi + ham ses (16 kHz) -> gerekceli JSON yorum. TEK nn.Module.

Icinde: BlazeFace yuz kirpici (torch) -> duygu7 ViT -> [goz tokenlari]; log-mel -> Whisper encoder -> [kulak tokenlari];
Whisper decoder (transkript); baglanti MLP'leri; yardimci basliklar (sinif, z-isaretler, kisilik[kapali]); Qwen2.5 + LoRA.
Disarida ayri model / olcum script'i yok. Tek config.json + safetensors; trust_remote_code ile yuklenir.

Girdi birimi "parca" (~3 sn): 8 kare + 16 kHz ses. on_hesapla() donmus ViT/Whisper-encoder ciktilarini uretir; egitimde
onbellege alinabilir (uctan uca ile esdeger, cunku bu modulller donmus) ve cikarimda ayni kod calisir.
"""
import json
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (PreTrainedModel, Qwen2Config, Qwen2ForCausalLM, ViTConfig, ViTModel, WhisperConfig,
                          WhisperForConditionalGeneration, GenerationConfig, LogitsProcessor, LogitsProcessorList)

from .blazeface import BlazeFace
from .configuration_davranis import DavranisConfig, BAS_GRUPLARI
from . import anlam_kurallari                 # uyum/anlam/ruh_hali/tutum/yorum: egitim hedefi (hedef_v2) ve ogretmenle AYNI islevler (v2/veri/anlam_cozumleme.json)
from .olcum_kurallari import GENEL_VARSAYILAN, bas_olcumleri, tablo_yukle, tekrar_kaliplari, kalip_dizisi_metni

YOK_IFADE = "ölçülmüş belirgin işaret yok"
KANIT_DIZI_LIT = '"kanitlar":'              # zorlanan kanit dizisinden hemen once gelen sabit parca
OLASI_LIT = '"olasi_isaret":'                # zorlanan olasi_isaret dizisinden hemen once gelen sabit parca (kodla doldurulur; kesin dille anilamaz)
KALIP_LIT = '"tekrar_eden_kaliplar":'       # zorlanan kalip dizisinden hemen once gelen sabit parca (dizi KURALLA hesaplanir: olcum_kurallari.tekrar_kaliplari)
KANIT_LIT = '"olcum":"'                       # kanit ifadesinden hemen once gelen sabit parca (kendi basina tokenlenir)
HEDEF_ISARET = ("\x01", "\x02")
ZORLA_ISARET = ("\x03", "\x04")             # modelin forward'inda KOPYALANAN bolge (kanit dizisi); LLM uretmez, kayip yok               # hedef metinde ifadeyi saran isaretciler (egitim etiketi uretiminde)
UYUM_DEGERLERI = ("uyumlu", "kismen", "uyumsuz", "degerlendirilemedi")
GIRDI_DEGERLERI = ("goruntu+ses", "sadece_goruntu", "sadece_ses", "girdi_yok")


# ----------------------------------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale


HEDEF_MODULLER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_ekle(llm, r, alpha, dropout):
    for _, mod in list(llm.model.layers.named_modules()):
        for ad in HEDEF_MODULLER:
            alt = getattr(mod, ad, None)
            if isinstance(alt, nn.Linear):
                setattr(mod, ad, LoRALinear(alt, r, alpha, dropout))


_LORA_ANAHTAR = re.compile(r"(\.(?:q|k|v|o|gate|up|down)_proj)\.(weight|bias)$")


def llm_anahtar_uyarla(sd):
    """Duz Qwen state_dict anahtarlari -> LoRALinear sarmali anahtarlar (.base.)."""
    return {_LORA_ANAHTAR.sub(r"\1.base.\2", k): v for k, v in sd.items()}


VARSAYILAN_KP = [[0.32, 0.40], [0.68, 0.40], [0.50, 0.58], [0.50, 0.75], [0.08, 0.50], [0.92, 0.50]]   # yuz bulunamazsa tipik yerlesim


def bolge_maskeleri(kp, grid=14):
    """kp [K,6,2] (kirpinti-normalize) -> [K,3,grid*grid] agirliklar (agiz, gozler, kaslar); ViT yama izgarasi uzerinde, satir-normalize.
    Bolgeler goz mesafesi d'ye oranli: agiz (+-0.55d, +-0.35d), gozler (her goz +-0.35d, +-0.22d), kaslar (goz ustu 0.18d-0.60d)."""
    K = kp.shape[0]
    c = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    gy, gx = torch.meshgrid(c, c, indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    cikti = torch.zeros(K, 3, grid * grid)
    for i in range(K):
        er, el, _, ag = kp[i, 0], kp[i, 1], kp[i, 2], kp[i, 3]
        d = float((er - el).norm().clamp(min=0.12))
        agiz = ((gx - ag[0]).abs() <= 0.55 * d) & ((gy - ag[1]).abs() <= 0.35 * d)
        goz = torch.zeros_like(agiz)
        kas = torch.zeros_like(agiz)
        for e in (er, el):
            goz |= ((gx - e[0]).abs() <= 0.35 * d) & ((gy - e[1]).abs() <= 0.22 * d)
            kas |= ((gx - e[0]).abs() <= 0.40 * d) & (gy >= e[1] - 0.60 * d) & (gy <= e[1] - 0.18 * d)
        for j, m_ in enumerate((agiz, goz, kas)):
            if not m_.any():                          # bolge izgaraya dusmediyse en yakin yama
                m_ = torch.zeros_like(agiz)
                ref = {0: ag, 1: er, 2: er - torch.tensor([0.0, 0.3 * d])}[j]
                m_[int(((gx - ref[0]) ** 2 + (gy - ref[1]) ** 2).argmin())] = True
            cikti[i, j] = m_.float() / m_.float().sum()
    return cikti


def geometri_ozellikleri(kp):
    """kp [K,6,2] -> [K,17]: 12 normalize nokta + goz mesafesi, agiz-burun/goz orani, burun yatay kaymasi (yaw vekili),
    goz hatti egimi (roll), burun-goz dikey orani (pitch vekili)."""
    er, el, bu, ag = kp[:, 0], kp[:, 1], kp[:, 2], kp[:, 3]
    d = (er - el).norm(dim=-1).clamp(min=0.05)
    orta = (er + el) / 2
    return torch.cat([kp.reshape(kp.shape[0], 12), d[:, None], ((ag - bu).norm(dim=-1) / d)[:, None], ((bu[:, 0] - orta[:, 0]) / d)[:, None],
                      torch.atan2(el[:, 1] - er[:, 1], el[:, 0] - er[:, 0] + 1e-6)[:, None], ((bu[:, 1] - orta[:, 1]) / d)[:, None]], -1)


GEO_D = 17


# ----------------------------------------------------------------------------------------------- yuz kirpici

class YuzKirpici(nn.Module):
    """BlazeFace (128x128 letterbox) ile her karede yuzu bulur, %pay genisletip 224x224 gri-3-kanal kirpinti uretir.
    Yuz bulunamayan karede: diger karelerin medyan kutusu; hic yuz yoksa merkez kare kirpma (bulundu=False)."""

    def __init__(self):
        super().__init__()
        self.blaze = BlazeFace()

    @torch.no_grad()
    def _tespit(self, kare, min_skor):
        """kare [K,3,H,W] float 0..255 -> (kutular [K] (x0,y0,x1,y1 orijinal piksel | None), noktalar [K] ([6,2] piksel | None), H, W). forward'in tespit adimi."""
        K, _, H, W = kare.shape
        s = 128.0 / max(H, W)
        nh, nw = max(int(round(H * s)), 1), max(int(round(W * s)), 1)
        kucuk = F.interpolate(kare, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
        py, px = (128 - nh) // 2, (128 - nw) // 2
        tuval = kare.new_zeros(K, 3, 128, 128)
        tuval[:, :, py:py + nh, px:px + nw] = kucuk
        det = self.blaze.kutular(tuval / 127.5 - 1.0, min_skor)
        kutular, noktalar = [], []
        for kutu, _, nk in det:
            if kutu is None:
                kutular.append(None)
                noktalar.append(None)
                continue
            ymin, xmin, ymax, xmax = (kutu * 128.0).tolist()
            kutular.append([(xmin - px) / s, (ymin - py) / s, (xmax - px) / s, (ymax - py) / s])
            nk = nk.detach().cpu()
            noktalar.append(((nk * 128.0) - torch.tensor([px, py], dtype=nk.dtype)) / s)          # orijinal piksel (x,y)
        return kutular, noktalar, H, W

    @torch.no_grad()
    def kutu_olc(self, kare, min_skor=0.5):
        """Yuz kutusu olcumu (bas_geri_cekilme icin): kare [K,3,H,W] -> (kutu [K,4] piksel x0,y0,x1,y1 (bulunamayan: NaN), bulundu [K] bool, H, W)."""
        kutular, _, H, W = self._tespit(kare, min_skor)
        kutu = torch.tensor([b if b is not None else [float("nan")] * 4 for b in kutular], dtype=torch.float32)
        return kutu, torch.tensor([b is not None for b in kutular]), H, W

    @torch.no_grad()
    def forward(self, kare, boyut=224, min_skor=0.5, pay=0.15, kp=False):
        """kare: [K,3,H,W] float 0..255 (RGB). Dondurur (kirpinti [K,3,boyut,boyut] -1..1 gri-RGB, bulundu [K] bool)
        ve kp=True ise ayrica anahtar noktalar [K,6,2] (kirpinti icinde 0..1: sag goz, sol goz, burun, agiz, sag kulak, sol kulak)."""
        from torchvision.ops import roi_align
        K = kare.shape[0]
        kutular, noktalar, H, W = self._tespit(kare, min_skor)
        bulundu = torch.tensor([b is not None for b in kutular])
        var = [b for b in kutular if b is not None]
        if var:
            yedek = torch.tensor(var).median(0).values.tolist()
        else:
            yk = 0.7 * min(H, W)
            yedek = [(W - yk) / 2, (H - yk) / 2, (W + yk) / 2, (H + yk) / 2]
        rois, kp_kirp = [], []
        for i, b in enumerate(kutular):
            x0, y0, x1, y1 = b if b is not None else yedek
            w, h = x1 - x0, y1 - y0
            x0, y0, x1, y1 = x0 - pay * w, y0 - pay * h, x1 + pay * w, y1 + pay * h
            x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(W)), min(y1, float(H))
            if x1 - x0 < 2 or y1 - y0 < 2:
                x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
            rois.append([float(i), x0, y0, x1, y1])
            if noktalar[i] is not None:
                kp_kirp.append(((noktalar[i] - torch.tensor([x0, y0])) / torch.tensor([x1 - x0, y1 - y0])).clamp(-0.2, 1.2))
            else:
                kp_kirp.append(None)
        gri = (0.299 * kare[:, 0:1] + 0.587 * kare[:, 1:2] + 0.114 * kare[:, 2:3])
        kirp = roi_align(gri, torch.tensor(rois, dtype=gri.dtype, device=gri.device), output_size=boyut,
                         spatial_scale=1.0, sampling_ratio=2, aligned=True)
        kirp = (kirp / 255.0 - 0.5) / 0.5
        kirp = kirp.expand(-1, 3, -1, -1).contiguous()
        if not kp:
            return kirp, bulundu
        var_kp = [k_ for k_ in kp_kirp if k_ is not None]
        yedek_kp = torch.stack(var_kp).median(0).values if var_kp else torch.tensor(VARSAYILAN_KP)
        return kirp, bulundu, torch.stack([k_ if k_ is not None else yedek_kp for k_ in kp_kirp])


# ----------------------------------------------------------------------------------------------- olcum basligi

class OlcumBasligi(nn.Module):
    """Olculen isaretler (z) ve guven (P(z>1.5), P(z<-1.5)). Girdi: ViT CLS (kare ort.), bolgesel yama havuzlari (agiz/gozler/kaslar),
    BlazeFace 6 nokta geometrisi (ort.+std), Whisper kulak ozellikleri (ort.+std), modalite bayraklari. Eksik modalite sifirlanir + bayrak.
    Donmus ozelliklerle LLM'siz on-egitimlenir (asama 0); LLM asamasinda dondurulur."""

    def __init__(self, dv, dw, n_isaret, gizli=1024, dropout=0.15):
        super().__init__()
        self.n_isaret = n_isaret
        girdi = dv * 4 + GEO_D * 2 + dw * 2 + 2
        self.norm = nn.LayerNorm(girdi)
        self.govde = nn.Sequential(nn.Linear(girdi, gizli), nn.GELU(), nn.Dropout(dropout), nn.Linear(gizli, gizli), nn.GELU(), nn.Dropout(dropout))
        self.z = nn.Linear(gizli, n_isaret)
        self.p = nn.Linear(gizli, 2 * n_isaret)       # [yukari (z>1.5) | asagi (z<-1.5)]

    def forward(self, goz, bolge, geo, kulak, goz_var, ses_var, pop_goz, pop_bolge, pop_kulak):
        """goz [B,8,dv]; bolge [B,8,3,dv]; geo [B,8,GEO_D]; kulak [B,16,dw]; *_var [B] bool. Dondurur (z [B,S], p_logit [B,2S])."""
        gv, sv = goz_var.to(goz.device).float()[:, None], ses_var.to(goz.device).float()[:, None]
        cls = (goz - pop_goz).mean(1) * gv
        bol = ((bolge - pop_bolge).mean(1)).flatten(1) * gv
        g_ort, g_std = geo.mean(1) * gv, geo.std(1) * gv
        k_ort, k_std = (kulak - pop_kulak).mean(1) * sv, kulak.std(1) * sv
        h = self.govde(self.norm(torch.cat([cls, bol, g_ort, g_std, k_ort, k_std, gv, sv], -1)))
        return self.z(h), self.p(h)


def istem_olcum_satirlari(olcum_c, olasi_c):
    """LLM istemindeki 'Ölçümler:' / 'Olası işaretler:' satirlari (parca basina). olay_ham ile AYNI yokluk filtresi uygulanir: hareketin YOKLUGUNU anlatan isaretler
    ('göz çevresi gevşek' vb.) istemde de GORUNMEZ -> LLM, yapilandirilmis kanit dizisinde olmayan bir ifadeyi anamaz (E0 halusinasyon vakasi). Dondurur (ol, oi)."""
    yok_ = anlam_kurallari.yokluk_isaretleri()
    ol_ = [(i, p) for i, p in olcum_c if str(i).lower() not in yok_]
    oi_ = [(i, p) for i, p in olasi_c if str(i).lower() not in yok_]
    ol = "; ".join(f"{ifade} (güven {p:.2f})" for ifade, p in ol_) or YOK_IFADE
    oi = "; ".join(f"{ifade} (güven {p:.2f})" for ifade, p in oi_) or "yok"
    return ol, oi


def olcum_listesi(z_hat, p_logit, mevcut, z_gruplari, ifadeler, esik=1.5, p_min=0.5, max_n=6, izinli=None, p_max=None):
    """Tek parca icin olculen isaretler: [(grup_idx, yon, ifade, |z|, guven)] |z|>esik ve P>=p_min (float ya da {grup:[p_yukari,p_asagi]}), |z|*P'ye gore siralanir.
    izinli: kanit sozlugunde kalan grup adlari (korelasyonu dusuk isaretler cikarilmis); None = hepsi."""
    p = torch.sigmoid(p_logit)
    S = len(z_gruplari)
    cikti = []
    for gi, g in enumerate(z_gruplari):
        if not bool(mevcut[gi]) or (izinli is not None and g not in izinli):
            continue
        for yon, ifade, pj in ((1, ifadeler[g][0], p[gi]), (-1, ifadeler[g][1], p[S + gi])):
            if ifade is None:
                continue
            zz = float(z_hat[gi]) * yon
            pm = (p_min.get(g, (0.5, 0.5))[0 if yon > 0 else 1]) if isinstance(p_min, dict) else p_min      # p_min: float ya da isaret/yon basina esik (esiksiz grup = geometrik baş/bakis: 0.5)
            px_ = None if p_max is None else ((p_max.get(g, (0.0, 0.0))[0 if yon > 0 else 1]) if isinstance(p_max, dict) else p_max)      # p_max: ust sinir (OLASI katman: p_min <= p < p_max)
            if zz > esik and float(pj) >= pm and (px_ is None or float(pj) < px_):
                cikti.append((gi, yon, ifade, zz, float(pj)))
    cikti.sort(key=lambda t: -(t[3] * t[4]))
    return cikti[:max_n]


# ----------------------------------------------------------------------------------------------- kanit trie

class KanitTrie:
    """40 kanit ifadesinin token dizilerinden onek agaci. Egitim cezasi ve cikarim kisitlamasi AYNI nesneyi kullanir."""

    def __init__(self, tok, isaret_ifadeleri, z_gruplari):
        self.ifadeler = []        # [(grup_index, yon, metin, token_ids)]
        for gi, g in enumerate(z_gruplari):
            for yon, ifade in zip((1, -1), isaret_ifadeleri[g]):
                if ifade:
                    self.ifadeler.append((gi, yon, ifade, tok(ifade, add_special_tokens=False)["input_ids"]))
        # her zaman izinli "olcum yok" ifadesi: yardimci baslik hicbir isaret olcmediginde kanit listesi bunu tasir
        self.yok_idx = len(self.ifadeler)
        self.ifadeler.append((-1, 0, YOK_IFADE, tok(YOK_IFADE, add_special_tokens=False)["input_ids"]))
        self.metin_idx = {e[2]: i for i, e in enumerate(self.ifadeler)}

    def izinli_kume(self, z_tah, mevcut, esik):
        """z_tah: [20] yardimci basligin z tahmini; mevcut: [20] bool. -> izinli ifade indeksleri kumesi."""
        return {i for i, (gi, yon, _, _) in enumerate(self.ifadeler)
                if gi < 0 or (mevcut[gi] and z_tah[gi] * yon > esik)}

    def yasak_tokenler(self, onek, izinli):
        """onek (ifadenin o ana kadarki token'lari) verildiginde, YALNIZCA izinsiz ifadelere giden sonraki token'lar."""
        uyan = [i for i, e in enumerate(self.ifadeler) if len(e[3]) > len(onek) and e[3][:len(onek)] == list(onek)]
        if not uyan:
            return set()
        if len(onek) > 0 and len({self.ifadeler[i][3][len(onek)] for i in uyan}) < 2:
            return set()                       # secim noktasi degil: ifade zaten secilmis (ilk token'dan sonra)
        izin_tok = {self.ifadeler[i][3][len(onek)] for i in uyan if i in izinli}
        return {self.ifadeler[i][3][len(onek)] for i in uyan if i not in izinli} - izin_tok


class KanitIslemci(LogitsProcessor):
    """Uretimde kanit ifadesi basladiginda (sabit '"olcum":"' parcasindan sonra) izinsiz ifadelere giden token'lari yasaklar.
    Izinli kume o kanitin ait oldugu parcanin yardimci-baslik olcumunden gelir (t alani); okunamazsa tum parcalarin birlesimi."""

    def __init__(self, model, trie, lit_ids, izinli_parca, parca_t0, bas_uzunluk):
        self.tok, self.trie, self.lit = model.tok, trie, list(lit_ids)
        self.izinli_parca, self.parca_t0, self.bas = izinli_parca, parca_t0, bas_uzunluk
        self.birlesim = set().union(*izinli_parca) if izinli_parca else set()

    def __call__(self, input_ids, scores):
        gen = input_ids[0, self.bas:].tolist() if input_ids.shape[1] > self.bas else []
        # son LIT konumunu bul
        n = len(self.lit)
        son = -1
        for i in range(len(gen) - n, -1, -1):
            if gen[i:i + n] == self.lit:
                son = i
                break
        if son < 0:
            return scores
        onek = gen[son + n:]
        if len(onek) > 0 and self.tok.decode(onek[-1:]).strip().endswith('"'):
            return scores                                   # ifade bitti
        once = self.tok.decode(gen[max(0, son - 80):son])
        m = list(re.finditer(r'"t":"([\d.]+)-', once))
        izinli = self.birlesim
        if m:
            t0 = float(m[-1].group(1))
            k = min(range(len(self.parca_t0)), key=lambda c: abs(self.parca_t0[c] - t0))
            izinli = self.izinli_parca[k]
        yasak = self.trie.yasak_tokenler(onek, izinli)
        if yasak:
            scores[:, list(yasak)] = -float("inf")
        return scores


# ----------------------------------------------------------------------------------------------- hedef metin parcalama

class ZorlaIslemci(LogitsProcessor):
    """Uretim sabit marker token'larini ('"kanitlar":') gordugunde, kanit dizisinin token'larini BIRER BIRER zorlar (digerleri -inf).
    Dizi bitince serbest birakir. input_ids yalniz uretilen token'lari tasir (inputs_embeds ile generate)."""

    def __init__(self, marker_ids, zorla_ids):
        self.marker, self.zorla = list(marker_ids), list(zorla_ids)

    def __call__(self, input_ids, scores):
        gen = input_ids[0].tolist() if input_ids.shape[1] else []
        n = len(self.marker)
        son = -1
        for i in range(len(gen) - n, -1, -1):
            if gen[i:i + n] == self.marker:
                son = i
                break
        if son < 0:
            return scores
        k = len(gen) - (son + n)                    # marker'dan beri uretilen/zorlanan token sayisi
        if k < len(self.zorla):
            hedef = self.zorla[k]
            maske = torch.full_like(scores, -float("inf"))
            maske[:, hedef] = 0
            return scores + maske
        return scores


class KalipZorlaIslemci(LogitsProcessor):
    """'"tekrar_eden_kaliplar":' sabit parcasindan sonra diziyi KURALLA hesaplayip token token zorlar (LLM uretmez). Parcalarin 'uyum' degerleri,
    LLM'in o ana kadar yazdigi zaman_cizelgesi'nden okunur (JSON'da zaman_cizelgesi olay'dan once gelir); sayilari eksik/bozuksa uyum bilinmiyor sayilir.
    Ayni islev egitim etiketinde de kullanilir (hedef_v2.olay_ham -> olcum_kurallari.tekrar_kaliplari)."""

    def __init__(self, tok, marker_ids, olcum_parca, konusma_parca):
        self.tok, self.marker = tok, list(marker_ids)
        self.olcum_parca, self.konusma_parca = olcum_parca, konusma_parca
        self._onbellek = {}

    def hesapla(self, once_metni):
        uyum = re.findall(r'"uyum"\s*:\s*"(uyumlu|kismen|uyumsuz|degerlendirilemedi)"', once_metni)
        uyum = uyum if len(uyum) == len(self.olcum_parca) else None
        return kalip_dizisi_metni(tekrar_kaliplari(self.olcum_parca, uyum, self.konusma_parca))

    def __call__(self, input_ids, scores):
        gen = input_ids[0].tolist() if input_ids.shape[1] else []
        n = len(self.marker)
        son = -1
        for i in range(len(gen) - n, -1, -1):
            if gen[i:i + n] == self.marker:
                son = i
                break
        if son < 0:
            return scores
        if son not in self._onbellek:
            self._onbellek[son] = self.tok(self.hesapla(self.tok.decode(gen[:son])), add_special_tokens=False)["input_ids"]
        zorla = self._onbellek[son]
        k = len(gen) - (son + n)
        if k < len(zorla):
            maske = torch.full_like(scores, -float("inf"))
            maske[:, zorla[k]] = 0
            return scores + maske
        return scores


def kanit_dizisi_metni(ts, olcum):
    """Modelin forward'inda kopyalanan 'kanitlar' dizisi (kompakt JSON). Olcum yoksa tek 'olcum yok' kaydi. hedef_v2.kanit_dizisi_metni ile ayni."""
    giris = [{"t": t, "olcum": ifade, "guven": round(float(p), 2)} for t, ol in zip(ts, olcum) for ifade, p in ol]
    if not giris:
        giris = [{"t": ts[0], "olcum": YOK_IFADE}]
    return json.dumps(giris, ensure_ascii=False, separators=(",", ":"))


def olasi_dizisi_metni(ts, olasi):
    """Forward'da KOPYALANAN 'olasi_isaret' dizisi: [{t, olcum, guven}] (bos olabilir -> '[]'). hedef_v2.olasi_dizisi_metni ile ayni bicim."""
    return json.dumps([{"t": t, "olcum": ifade, "guven": round(float(p), 2)} for t, ol in zip(ts, olasi) for ifade, p in ol], ensure_ascii=False, separators=(",", ":"))


def hedef_parcala(metin_isaretli):
    """Isaretli hedef metni -> [("m", metin) | ("lit", '"olcum":"') | ("kanit", ifade) | ("lit2", '"kanitlar":') | ("zorla", metin)] ve duz metin.
    \\x01ifade\\x02 = LLM'in uretecegi kanit ifadesi (eski yontem, kanit-ceza kaybi); \\x03metin\\x04 = modelin forward'inda ZORLA kopyalanan
    bolge (kanit dizisi: kayip YOK, uretimde zorlanir)."""
    a, b = HEDEF_ISARET
    za, zb = ZORLA_ISARET
    parcalar = []
    for parca in re.split(f"({za}[^{zb}]*{zb})", metin_isaretli):
        if parca.startswith(za):
            parcalar.append(("zorla", parca[1:-1]))
            continue
        for parca2 in re.split(f"({a}[^{b}]*{b})", parca):
            if parca2.startswith(a):
                parcalar.append(("kanit", parca2[1:-1]))
            elif parca2:
                for lit, ad in ((KANIT_LIT, "lit"), (KANIT_DIZI_LIT, "lit2"), (KALIP_LIT, "lit3"), (OLASI_LIT, "lit4")):
                    if parca2.endswith(lit):
                        if parca2[:-len(lit)]:
                            parcalar.append(("m", parca2[:-len(lit)]))
                        parcalar.append((ad, lit))
                        break
                else:
                    parcalar.append(("m", parca2))
    duz = metin_isaretli
    for ch in (a, b, za, zb):
        duz = duz.replace(ch, "")
    return parcalar, duz


def sema_gecerli(j):
    """Cikti JSON'u beklenen semaya uyuyor mu?"""
    try:
        zc, o = j["zaman_cizelgesi"], j["olay"]
        if not (isinstance(zc, list) and zc):
            return False
        for e in zc:
            if not all(k in e for k in ("t", "soylenen", "davranis", "yorum", "uyum")) or e["uyum"] not in UYUM_DEGERLERI:
                return False
        need = ("girdi", "ruh_hali", "duygu", "guven", "soz_davranis_uyumu", "tutum", "tekrar_eden_kaliplar",
                "kisilik_izlenimi", "kanitlar", "genel_yorum")
        if not all(k in o for k in need):
            return False
        if "olasi_isaret" in o and not (isinstance(o["olasi_isaret"], list) and all(isinstance(k, dict) and "t" in k and "olcum" in k and "guven" in k for k in o["olasi_isaret"])):
            return False
        return isinstance(o["soz_davranis_uyumu"], dict) and "durum" in o["soz_davranis_uyumu"] and "skor" in o["soz_davranis_uyumu"] \
            and isinstance(o["kanitlar"], list) and all("t" in k and "olcum" in k for k in o["kanitlar"])
    except Exception:
        return False


def json_ayikla(metin):
    """Uretilen metinden ilk tam JSON nesnesini ayiklar; yoksa None."""
    bas = metin.find("{")
    if bas < 0:
        return None
    derin, ic_str, kacis = 0, False, False
    for i in range(bas, len(metin)):
        c = metin[i]
        if ic_str:
            if kacis:
                kacis = False
            elif c == "\\":
                kacis = True
            elif c == '"':
                ic_str = False
        elif c == '"':
            ic_str = True
        elif c == "{":
            derin += 1
        elif c == "}":
            derin -= 1
            if derin == 0:
                try:
                    return json.loads(metin[bas:i + 1])
                except Exception:
                    return None
    return None


# ----------------------------------------------------------------------------------------------- model

class _YokLLM(nn.Module):
    """Ozellik modu yer tutucusu (LLM yok): on_hesapla ve olcum basligi icin yeterli; girdi_kur/uret KULLANILAMAZ."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 8})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()

    def get_input_embeddings(self):
        return nn.Embedding(1, 8)


class DavranisModel(PreTrainedModel):
    config_class = DavranisConfig
    base_model_prefix = "davranis"
    main_input_name = "kareler"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"llm.lm_head.weight": "llm.model.embed_tokens.weight",
                          "whisper.proj_out.weight": "whisper.model.decoder.embed_tokens.weight"}

    def __init__(self, config: DavranisConfig, llm=None):
        super().__init__(config)
        c = config
        self.yuz = YuzKirpici()
        self.vit = ViTModel(ViTConfig(**c.vit_config), add_pooling_layer=False)
        self.whisper = WhisperForConditionalGeneration(WhisperConfig(**c.whisper_config))
        if c.whisper_generation_config:
            self.whisper.generation_config = GenerationConfig(**c.whisper_generation_config)
        # llm verilirse (4-bit nf4 QLoRA yolu: parcalardan(llm_nf4=True)) ayri fp32 Qwen kurulmaz
        self.llm = llm if llm is not None else Qwen2ForCausalLM(Qwen2Config(**c.llm_config))
        lora_ekle(self.llm, c.lora_r, c.lora_alpha, c.lora_dropout)
        H = self.llm.config.hidden_size
        dv, dw = self.vit.config.hidden_size, self.whisper.config.d_model
        self.goz_mlp = nn.Sequential(nn.Linear(dv, H), nn.GELU(), nn.Linear(H, H))
        self.kulak_mlp = nn.Sequential(nn.Linear(dw, H), nn.GELU(), nn.Linear(H, H))
        self.modalite = nn.Parameter(torch.randn(2, H) * 0.02)
        self.poz_goz = nn.Parameter(torch.randn(c.n_goz, H) * 0.02)
        self.poz_kulak = nn.Parameter(torch.randn(c.n_kulak, H) * 0.02)
        self.goz_yok = nn.Parameter(torch.randn(c.n_goz, H) * 0.02)       # goruntu yok/sifir -> ogrenilmis "yok" gomulmesi
        self.kulak_yok = nn.Parameter(torch.randn(c.n_kulak, H) * 0.02)   # ses yok/sessiz
        self.sinif_basligi = nn.Linear(H, len(c.siniflar))
        self.z_basligi = nn.Linear(H, len(c.z_gruplari))                 # ESKI (kullanilmiyor; geriye uyumluluk)
        self.n_ogrenilen = len([g for g in c.z_gruplari if g not in BAS_GRUPLARI])           # baş/bakis isaretleri OGRENILMEZ (geometrik)
        self.olcum_basligi = OlcumBasligi(dv, dw, self.n_ogrenilen)     # asama 0'da on-egitilir, LLM asamasinda DONUK
        self.kisilik_basligi = nn.Linear(H, len(c.kisilik_alanlari))      # kisilik_aktif=False iken egitilmez/kullanilmaz
        self.register_buffer("pop_goz_ort", torch.zeros(dv))              # populasyon normu (egitimden)
        self.register_buffer("pop_kulak_ort", torch.zeros(dw))
        self.register_buffer("pop_bolge_ort", torch.zeros(3, dv))
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor(feature_size=self.whisper.config.num_mel_bins)
        self.register_buffer("mel_filtre", torch.tensor(np.asarray(fe.mel_filters), dtype=torch.float32))   # [201, 80]
        self.register_buffer("mel_pencere", torch.hann_window(400))
        self.tok = None            # Qwen tokenizer (tokenizerlari_yukle / tokenizer_ayarla)
        self.whisper_tok = None
        self._trie = None
        self.post_init()

    def _init_weights(self, module):          # alt modullerin on-egitimli / ozel baslangiclarini bozma
        return

    # ------------------------------------------------------------------ kurulum yardimcilari
    @classmethod
    def parcalardan(cls, vit_dizini, whisper_adi="openai/whisper-small", llm_adi="Qwen/Qwen2.5-1.5B-Instruct",
                    blaze_dizini=None, llm_nf4=False, llm_dtype=None, **cfg_kw):
        """Onceden egitilmis parcalardan DavranisModel kurar (ViT: duygu7_final, Whisper, Qwen) ve BlazeFace agirligini yukler."""
        import os
        from transformers import ViTForImageClassification
        vit = ViTForImageClassification.from_pretrained(vit_dizini)
        wh = WhisperForConditionalGeneration.from_pretrained(whisper_adi)
        ozellik_modu = llm_adi is None          # yalniz on_hesapla/olcum basligi (ozellik onbellegi, asama 0): LLM yuklenmez
        if ozellik_modu:
            llm = _YokLLM()
        elif llm_nf4:                 # QLoRA: taban 4-bit nf4 (bitsandbytes), LoRA A/B fp32; GPU gerekir
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
            llm = AutoModelForCausalLM.from_pretrained(llm_adi, quantization_config=q, device_map={"": 0}, dtype=torch.float16)
        elif llm_dtype is not None:   # CIKARIM (canli/HF Space): taban dogrudan dusuk hassasiyette yuklenir (7B fp32 ~30 GB'a cikmaz); egitim yolu degismez
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(llm_adi, dtype=llm_dtype)
        else:
            llm = Qwen2ForCausalLM.from_pretrained(llm_adi, dtype=torch.float32)
        cfg = DavranisConfig(llm_adi=llm_adi or "", llm_nf4=llm_nf4, llm_config={} if ozellik_modu else llm.config.to_dict(), vit_config=vit.config.to_dict(),
                             whisper_config=wh.config.to_dict(),
                             whisper_generation_config=wh.generation_config.to_dict(), **cfg_kw)
        m = cls(cfg, llm=llm if (llm_nf4 or ozellik_modu or llm_dtype is not None) else None)
        m.vit.load_state_dict(vit.vit.state_dict())
        m.whisper.load_state_dict(wh.state_dict())
        if not llm_nf4 and not ozellik_modu and llm_dtype is None:
            m.llm.load_state_dict(llm_anahtar_uyarla(llm.state_dict()), strict=False)   # LoRA A/B eksik kalir (beklenen)
        if not ozellik_modu:
            m.llm.tie_weights()
        blaze_dizini = blaze_dizini or os.path.dirname(os.path.abspath(__file__))
        m.yuz.blaze.load_state_dict(torch.load(os.path.join(blaze_dizini, "blazeface.pth"), map_location="cpu"), strict=False)
        m.yuz.blaze.anchors.copy_(torch.tensor(np.load(os.path.join(blaze_dizini, "anchors.npy")), dtype=torch.float32))
        m.yuz.blaze.eval()
        return m

    def tokenizer_ayarla(self, llm_tok, whisper_tok=None):
        self.tok = llm_tok
        self.whisper_tok = whisper_tok
        self._trie = KanitTrie(llm_tok, self.config.isaret_ifadeleri, self.config.z_gruplari)
        return self

    def tokenizerlari_yukle(self, dizin):
        import os
        from transformers import AutoTokenizer
        wt = os.path.join(dizin, "whisper_tokenizer")
        return self.tokenizer_ayarla(AutoTokenizer.from_pretrained(os.path.join(dizin, "llm_tokenizer")),
                                     AutoTokenizer.from_pretrained(wt) if os.path.isdir(wt) else None)

    def tokenizerlari_kaydet(self, dizin):
        import os
        self.tok.save_pretrained(os.path.join(dizin, "llm_tokenizer"))
        if self.whisper_tok is not None:
            self.whisper_tok.save_pretrained(os.path.join(dizin, "whisper_tokenizer"))

    def asama0_ayarla(self):
        """ASAMA 0: yalniz olcum basligi egitilir (LLM/ViT/Whisper donuk, LLM'siz). Egitilebilir ad listesini dondurur."""
        for p_ in self.parameters():
            p_.requires_grad = False
        egit = []
        for n_, p_ in self.olcum_basligi.named_parameters():
            p_.requires_grad = True
            egit.append("olcum_basligi." + n_)
        return egit

    def egitilebilir_ayarla(self):
        """ViT, Whisper, BlazeFace, Qwen tabani DONMUS; egitilen: LoRA, baglanti MLP'leri, parametreler, basliklar."""
        for p in self.parameters():
            p.requires_grad = False
        egit = []
        for n, p in self.named_parameters():
            if n.startswith("llm.") and (n.endswith(".A") or n.endswith(".B")):
                p.requires_grad = True
            elif n.split(".")[0] in ("goz_mlp", "kulak_mlp", "modalite", "poz_goz", "poz_kulak", "goz_yok", "kulak_yok",
                                       "sinif_basligi") or (n.startswith("kisilik_basligi") and self.config.kisilik_aktif):
                p.requires_grad = True
            if p.requires_grad:
                egit.append(n)
        for mod in (self.yuz, self.vit, self.whisper):
            mod.eval()
        return egit

    # ------------------------------------------------------------------ ON HESAPLAMA (donmus ViT + Whisper encoder)
    def _logmel(self, ses):
        """ses: [B,S] float32 16 kHz -> Whisper log-mel [B,80,3000] (30 sn'ye sifir tamamlanir)."""
        n = 30 * self.config.ses_sr
        ses = ses[:, :n]
        ses = F.pad(ses, (0, n - ses.shape[1]))
        st = torch.stft(ses, 400, 160, window=self.mel_pencere.to(ses.device), return_complex=True)
        mag = st[..., :-1].abs() ** 2
        mel = self.mel_filtre.to(ses.device).t() @ mag
        lg = torch.clamp(mel, min=1e-10).log10()
        lg = torch.maximum(lg, lg.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (lg + 4.0) / 4.0

    @staticmethod
    def _parca_ortala(h, n_kare, n_parca):
        n = int(max(1, min(n_kare, h.shape[0])))
        out = []
        for i in range(n_parca):
            a = (i * n) // n_parca
            b = max(((i + 1) * n) // n_parca, a + 1)
            out.append(h[a:b].float().mean(0))
        return torch.stack(out)

    @torch.no_grad()
    def on_hesapla(self, kareler, ses, yuz_bozucu=None, ses_bozucu=None):
        """Parca listeleri uzerinde donmus ozellik cikarir.
        kareler: [parca] her biri [K,3,H,W] uint8/float 0..255 RGB ham video kareleri (ya da None = goruntu yok);
        ses: [parca] her biri 1-D float 16 kHz dalga (ya da None = ses yok).
        yuz_bozucu(list[HxWx3 uint8]) -> list ; ses_bozucu(np.float32 dalga) -> dalga: egitim augmentasyonu (kirpindi / dalga uzerinde).
        Dondurur dict: goz [B,8,dv], kulak [B,16,dw], goz_var [B], ses_var [B], yuz_bulunan [B] (bulunan kare sayisi)."""
        c = self.config
        dev = next(self.vit.parameters()).device
        B = len(kareler)
        dv, dw = self.vit.config.hidden_size, self.whisper.config.d_model
        goz = torch.zeros(B, c.n_goz, dv)
        bolge = torch.zeros(B, c.n_goz, 3, dv)
        geo = torch.zeros(B, c.n_goz, GEO_D)
        kutu_olcek = torch.full((B, c.n_goz), float("nan"))          # yuz kutusu genisligi / kare genisligi (bas_geri_cekilme); bulunamayan kare: NaN
        kulak = torch.zeros(B, c.n_kulak, dw)
        goz_var = torch.zeros(B, dtype=torch.bool)
        ses_var = torch.zeros(B, dtype=torch.bool)
        yuz_n = torch.zeros(B, dtype=torch.long)
        # --- goruntu
        for i, k in enumerate(kareler):
            if k is None or k.numel() == 0 or float(k.max()) == 0.0:
                continue
            k = k.to(dev).float()
            idx = torch.linspace(0, k.shape[0] - 1, c.n_kare).round().long() if k.shape[0] != c.n_kare else torch.arange(c.n_kare)
            k = k[idx.to(dev)]
            kirp, bulundu, kpn = self.yuz(k, c.yuz_boyut, c.yuz_min_skor, c.yuz_pay, kp=True)
            if yuz_bozucu is not None:
                u8 = ((kirp * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
                u8 = yuz_bozucu([x for x in u8])
                kirp = torch.from_numpy(np.stack(u8)).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5).to(dev)
            hs = self.vit(pixel_values=kirp.to(next(self.vit.parameters()).dtype)).last_hidden_state      # [K,1+196,dv]
            cls = hs[:, 0]
            maske = bolge_maskeleri(kpn.cpu()).to(hs.device, hs.dtype)                                  # [K,3,196]
            bolge[i] = torch.einsum("krp,kpd->krd", maske, hs[:, 1:]).float().cpu()
            geo[i] = geometri_ozellikleri(kpn.cpu().float())
            kb, kv, _, kW = self.yuz.kutu_olc(k, c.yuz_min_skor)
            kutu_olcek[i] = torch.where(kv, (kb[:, 2] - kb[:, 0]) / kW, torch.full_like(kb[:, 0], float("nan")))
            goz[i] = cls.float().cpu()
            goz_var[i] = True
            yuz_n[i] = int(bulundu.sum())
        # --- ses
        enc = self.whisper.model.encoder
        gec, dalgalar = [], []
        for i, s in enumerate(ses):
            if s is None or s.numel() == 0 or float(s.abs().max()) == 0.0 or \
                    float(torch.sqrt((s.float() ** 2).mean())) < c.ses_sessiz_rms:
                continue
            d = s.float().cpu().numpy()
            if ses_bozucu is not None:
                d = ses_bozucu(d)
            gec.append(i)
            dalgalar.append(torch.from_numpy(np.asarray(d, dtype=np.float32))[: 30 * c.ses_sr])
        if gec:
            uz = max(len(d) for d in dalgalar)
            toplu = torch.stack([F.pad(d, (0, uz - len(d))) for d in dalgalar]).to(dev)
            mel = self._logmel(toplu).to(next(enc.parameters()).dtype)
            h = enc(mel).last_hidden_state
            for j, i in enumerate(gec):
                kulak[i] = self._parca_ortala(h[j], math.ceil(len(dalgalar[j]) / c.ses_sr * 50), c.n_kulak).cpu()
                ses_var[i] = True
        return {"goz": goz, "kulak": kulak, "goz_var": goz_var, "ses_var": ses_var, "yuz_bulunan": yuz_n, "bolge": bolge, "geo": geo, "kutu_olcek": kutu_olcek}

    @torch.no_grad()
    def populasyon_normu_ayarla(self, goz_ort, kulak_ort, bolge_ort=None):
        if bolge_ort is not None:
            self.pop_bolge_ort.copy_(torch.as_tensor(bolge_ort, dtype=torch.float32))
        self.pop_goz_ort.copy_(torch.as_tensor(goz_ort, dtype=torch.float32))
        self.pop_kulak_ort.copy_(torch.as_tensor(kulak_ort, dtype=torch.float32))

    # ------------------------------------------------------------------ transkript (Whisper decoder)
    @torch.no_grad()
    def transkript_uret(self, ses, dil=None, max_yeni=64):
        """ses: [parca] 1-D float 16 kHz (None = ses yok). Whisper decoder ile Turkce/Ingilizce metin; sessiz -> ''."""
        assert self.whisper_tok is not None, "whisper tokenizer yuklenmedi"
        c = self.config
        dev = next(self.whisper.parameters()).device
        cikti = [""] * len(ses)
        gec = [i for i, s in enumerate(ses) if s is not None and s.numel() > 0 and
               float(torch.sqrt((s.float() ** 2).mean())) >= c.ses_sessiz_rms]
        if not gec:
            return cikti
        d = [ses[i].float().cpu()[: 30 * c.ses_sr] for i in gec]
        uz = max(len(x) for x in d)
        mel = self._logmel(torch.stack([F.pad(x, (0, uz - len(x))) for x in d]).to(dev))
        kw = {"task": "transcribe", "max_new_tokens": max_yeni}
        if dil:
            kw["language"] = dil
        ids = self.whisper.generate(input_features=mel.to(next(self.whisper.parameters()).dtype), **kw)
        for j, i in enumerate(gec):
            cikti[i] = self.whisper_tok.decode(ids[j], skip_special_tokens=True).strip()
        return cikti

    # ------------------------------------------------------------------ algi tokenlari
    def algi(self, goz, kulak, goz_var, ses_var, norm_goz=None, norm_kulak=None):
        """goz [B,8,dv], kulak [B,16,dw] -> [B,24,H]. Kisi/populasyon normu cikarilir; yok olan modalite "yok" gomulmesi alir."""
        ng = self.pop_goz_ort if norm_goz is None else norm_goz
        nk = self.pop_kulak_ort if norm_kulak is None else norm_kulak
        ng = ng.to(goz.device).view(-1, 1, goz.shape[-1]) if ng.dim() > 1 else ng.to(goz.device)
        nk = nk.to(kulak.device).view(-1, 1, kulak.shape[-1]) if nk.dim() > 1 else nk.to(kulak.device)
        g = self.goz_mlp(goz - ng) + self.modalite[0] + self.poz_goz
        k = self.kulak_mlp(kulak - nk) + self.modalite[1] + self.poz_kulak
        gv, sv = goz_var.to(goz.device).view(-1, 1, 1), ses_var.to(goz.device).view(-1, 1, 1)
        g = torch.where(gv, g, self.goz_yok.expand_as(g))
        k = torch.where(sv, k, self.kulak_yok.expand_as(k))
        return torch.cat([g, k], 1)

    def kafalar(self, algi):
        """algi [B,24,H] -> (sinif logit [B,8], z [B,20], kisilik [B,5])."""
        p = algi.float().mean(1)
        return self.sinif_basligi(p.to(self.sinif_basligi.weight.dtype)), self.z_basligi(p.to(self.z_basligi.weight.dtype)), \
            self.kisilik_basligi(p.to(self.kisilik_basligi.weight.dtype))

    def olcum_basligi_tahmin(self, ornek):
        """OGRENILEN olcum basligi: (z [C,S_ogrenilen], p_logit [C,2*S_ogrenilen]). ornek: goz/bolge/geo/kulak [C,...], goz_var, ses_var."""
        dev = self.pop_goz_ort.device
        C = ornek["goz"].shape[0]
        bolge = ornek.get("bolge")
        geo = ornek.get("geo")
        if bolge is None:
            bolge = torch.zeros(C, self.config.n_goz, 3, ornek["goz"].shape[-1])
        if geo is None:
            geo = torch.zeros(C, self.config.n_goz, GEO_D)
        return self.olcum_basligi(ornek["goz"].to(dev).float(), bolge.to(dev).float(), geo.to(dev).float(), ornek["kulak"].to(dev).float(),
                                  ornek["goz_var"], ornek["ses_var"], self.pop_goz_ort, self.pop_bolge_ort, self.pop_kulak_ort)

    def bas_tablosu(self):
        o = getattr(self.config, "bas_norm", None) or dict(GENEL_VARSAYILAN)
        return tablo_yukle(o)

    def bas_tahmin(self, ornek):
        """BAS/BAKIS isaretleri: GEOMETRIK (BlazeFace 6 nokta; ogrenilmez), kisi bazli z. Dondurur (z [C,4], mevcut [C,4] bool).
        ornek: geo [C,K,17], yuz_bulunan [C] (ops.), kisi (ops. str; tabloda yoksa genel norm), kutu_olcek [C,K] (ops.; yoksa bas_geri_cekilme mevcut degil)."""
        C = ornek["goz"].shape[0]
        geo = ornek.get("geo")
        if geo is None:
            return torch.zeros(C, len(BAS_GRUPLARI)), torch.zeros(C, len(BAS_GRUPLARI), dtype=torch.bool)
        yuz = ornek.get("yuz_bulunan")
        kisi = ornek.get("kisi")
        z, gec = bas_olcumleri(geo.float().cpu().numpy(), None if yuz is None else torch.as_tensor(yuz).cpu().numpy(), self.bas_tablosu(),
                               None if kisi is None else np.array([kisi] * C),
                               None if ornek.get("kutu_olcek") is None else torch.as_tensor(ornek["kutu_olcek"]).float().cpu().numpy())
        gv = torch.as_tensor(ornek["goz_var"]).cpu().bool()[:, None]
        return torch.from_numpy(z), torch.from_numpy(gec) & gv

    def olcum_tahmin(self, ornek):
        """TUM isaretler (config.z_gruplari sirasi): (z [C,S], p_logit [C,2S]). Ogrenilen 20 + geometrik baş/bakis isaretleri (p: mevcutsa ~1)."""
        z, pl = self.olcum_basligi_tahmin(ornek)
        zb, gec = self.bas_tahmin(ornek)
        dev = z.device
        S0 = z.shape[1]
        BIG = 20.0
        yukari = torch.where(gec, BIG, -BIG).to(dev, pl.dtype)
        asagi = torch.full_like(yukari, -BIG)
        return torch.cat([z, zb.to(dev, z.dtype)], -1), torch.cat([pl[:, :S0], yukari, pl[:, S0:], asagi], -1)

    def olcumler(self, ornek):
        """Parca basina KANIT katmani [[(ifade, guven)]] (olcumler_iki()[0])."""
        return self.olcumler_iki(ornek)[0]

    def olcumler_iki(self, ornek):
        """IKI KATMAN: (kanit [[(ifade, guven)]], olasi [[(ifade, guven)]]) parca basina.
        kanit: |z|>kanit_esik ve P>=kanit_p_min (kesin dille yazilir); olasi: |z|>kanit_esik ve olasi_p_min <= P < kanit_p_min (yalniz temkinli dille).
        Geometrik baş/bakis isaretleri P=1 -> KANIT katmaninda. ornek['olcum_override'] / ['olasi_override'] (egitimde out-of-fold) varsa onlar kullanilir."""
        if ornek.get("olcum_override") is not None:
            ov = ornek["olcum_override"]
            return ov, (ornek.get("olasi_override") if ornek.get("olasi_override") is not None else [[] for _ in ov])
        c = self.config
        with torch.no_grad():
            z, pl = self.olcum_tahmin(ornek)
        ko = ornek.get("kutu_olcek")
        kutu_var = None if ko is None else (torch.isfinite(torch.as_tensor(ko)).float().mean(1) >= 0.5)
        maske = self.mevcut_maske(ornek["goz_var"], ornek["ses_var"], kutu_var)
        izinli = set(c.kanit_sozlugu) if c.kanit_sozlugu else None
        if izinli is not None:                       # E0: MediaPipe ile r=0.15 (yuz genisligi oranı) -> bas_geri_cekilme kanit olarak sunulmaz
            izinli -= set(getattr(c, "guvenilmez_gruplar", ("bas_geri_cekilme",)))
        olasi_min = getattr(c, "olasi_p_min", 0.7)
        kanit, olasi = [], []
        for i in range(z.shape[0]):
            zi, pi, mi = z[i].float().cpu(), pl[i].float().cpu(), maske[i]
            kanit.append([(x[2], x[4]) for x in olcum_listesi(zi, pi, mi, c.z_gruplari, c.isaret_ifadeleri, c.kanit_esik, c.kanit_p_min, c.olcum_max, izinli)])
            kn_ad = {x[0] for x in kanit[-1]}
            olasi.append([(x[2], x[4]) for x in olcum_listesi(zi, pi, mi, c.z_gruplari, c.isaret_ifadeleri, c.kanit_esik, olasi_min,
                                                              getattr(c, "olasi_max", 3), izinli, p_max=c.kanit_p_min) if x[2] not in kn_ad])      # KANITTA olan isaret olasi listesinde OLAMAZ
        return kanit, olasi

    def mevcut_maske(self, goz_var, ses_var, kutu_var=None):
        """[B,20] bool: o parcada olculebilir z gruplari."""
        ses_g = torch.tensor([g in self.config.ses_gruplari for g in self.config.z_gruplari])
        m = torch.where(ses_g[None, :], ses_var[:, None].cpu(), goz_var[:, None].cpu())
        for j, g in enumerate(self.config.z_gruplari):
            if g == "bas_geri_cekilme":              # yuz kutusu olcegi (kutu_olcek) varsa olculur, yoksa degil
                m[:, j] = False if kutu_var is None else (torch.as_tensor(kutu_var).cpu().bool() & goz_var.cpu().bool())
        return m

    # ------------------------------------------------------------------ girdi kurma (egitim + cikarim ortak)
    def _ids(self, metin):
        return self.tok(metin, add_special_tokens=False)["input_ids"]

    def girdi_kur(self, ornek, hedef_parcalari=None):
        """Tek ornek: {goz,kulak,bolge,geo,goz_var,ses_var [C,...], t [C str], transkript [C str], norm_goz/norm_kulak (ops.),
        olcum_override (ops.)}. Her parcanin istemine 'Olcumler:' satiri (modelin KENDI olcum basligi) eklenir; cikti 'kanitlar' dizisi bu
        listeden BIREBIR kopyalanir (LLM uretmez). Dondurur (E [L,H], etiket [L] (-100: prompt/zorlanan), algi [C,24,H],
        bilgi{olcum, kanit_metni, zorla_ids})."""
        emb = self.llm.get_input_embeddings()
        dev = emb.weight.device
        C = ornek["goz"].shape[0]
        algi = self.algi(ornek["goz"].to(dev), ornek["kulak"].to(dev), ornek["goz_var"], ornek["ses_var"],
                         ornek.get("norm_goz"), ornek.get("norm_kulak"))
        olcum, olasi = self.olcumler_iki(ornek)
        kanit_metni = kanit_dizisi_metni(ornek["t"], olcum)
        olasi_metni = olasi_dizisi_metni(ornek["t"], olasi)
        dt = emb.weight.dtype
        e = lambda ids: emb(torch.tensor(ids, device=dev))
        parca = [e(self._ids(f"<|im_start|>system\n{self.config.sistem}<|im_end|>\n<|im_start|>user\n"))]
        for c in range(C):
            parca.append(e(self._ids(f"[{ornek['t'][c]} sn] ")))
            parca.append(algi[c].to(dt))
            soz = ornek["transkript"][c] if ornek["transkript"][c] else ("(ses yok)" if not bool(ornek["ses_var"][c]) else "(konuşma yok)")
            ol, oi = istem_olcum_satirlari(olcum[c], olasi[c])
            parca.append(e(self._ids(f" Söylenen: {soz}\n Ölçümler: {ol}\n Olası işaretler (yalnız temkinli anılır): {oi}\n")))
        parca.append(e(self._ids("Bu kişinin davranışını değerlendir.<|im_end|>\n<|im_start|>assistant\n")))
        L0 = sum(p_.shape[0] for p_ in parca)
        etiket = [-100] * L0
        if hedef_parcalari is not None:
            for tur, metin in hedef_parcalari:
                ids = self._ids(metin)
                parca.append(e(ids))
                etiket += [-100] * len(ids) if tur == "zorla" else ids          # zorlanan kanit dizisinde KAYIP YOK
            son = self.tok.convert_tokens_to_ids("<|im_end|>")
            parca.append(e([son]))
            etiket.append(son)
        bilgi = {"olcum": olcum, "olasi": olasi, "kanit_metni": kanit_metni, "olasi_metni": olasi_metni, "zorla_ids": self._ids(kanit_metni),
                 "olasi_ids": self._ids(olasi_metni)}
        return torch.cat(parca, 0), torch.tensor(etiket, device=dev), algi, bilgi

    def kayip(self, ornekler, aux_w=0.5, kanit_w=0.0):
        """ornekler: [{girdi_kur alanlari, hedef_parcalari, y [C]}]. Terimler: LM (zorlanan kanit dizisi HARIC) + aux sinif CE.
        z/kanit: olcum basligi asama 0'da egitilir ve burada donuktur; kanit dizisi modelin forward'inda kopyalandigi icin
        kanit-ceza kaybi KALDIRILDI (kanit_w yoksayilir; metrik anahtarlari geriye uyum icin 0 doner)."""
        dev = self.llm.get_input_embeddings().weight.device
        seqler, etiketler, algilar = [], [], []
        for o in ornekler:
            E, Y, algi, _ = self.girdi_kur(o, o["hedef_parcalari"])
            seqler.append(E); etiketler.append(Y); algilar.append(algi)
        L = max(s_.shape[0] for s_ in seqler)
        n, H = len(seqler), seqler[0].shape[1]
        E = torch.zeros(n, L, H, dtype=seqler[0].dtype, device=dev)
        M = torch.zeros(n, L, dtype=torch.long, device=dev)
        Y = torch.full((n, L), -100, dtype=torch.long, device=dev)
        for j, s_ in enumerate(seqler):
            E[j, :s_.shape[0]], M[j, :s_.shape[0]], Y[j, :s_.shape[0]] = s_, 1, etiketler[j]
        h = self.llm.model(inputs_embeds=E, attention_mask=M).last_hidden_state
        sec = Y[:, 1:] != -100
        logits = self.llm.lm_head(h[:, :-1][sec]).float()
        lm = F.cross_entropy(logits, Y[:, 1:][sec])
        ce_l = []
        for j, o in enumerate(ornekler):
            s_lg, _, _ = self.kafalar(algilar[j])
            ce_l.append(F.cross_entropy(s_lg.float(), o["y"].to(dev), reduction="sum"))
        ce = torch.stack(ce_l).sum() / sum(len(o["y"]) for o in ornekler)
        toplam = lm + aux_w * ce
        return toplam, {"lm": float(lm), "ce": float(ce), "mse": 0.0, "kanit_ceza": 0.0, "kanit_pozisyon": 0, "kanit_ihlal": 0}

    def olcum_kayip(self, ornekler, esik=1.5, p_w=0.5):
        """ASAMA 0 kaybi (LLM'siz): olcum basligi. ornekler: [{goz,bolge,geo,kulak,goz_var,ses_var [C,...], z [C,S] (kirpilmis), z_ham [C,S]}].
        Terimler: maskeli MSE (z) + BCE (P(z>esik), P(z<-esik)). Kullanilamayan modalitenin isaretleri maskelenir."""
        dev = self.pop_goz_ort.device
        z_ort, bce_ort, n = 0.0, 0.0, 0
        for o in ornekler:
            z_hat, pl = self.olcum_basligi_tahmin(o)
            maske = self.mevcut_maske(o["goz_var"], o["ses_var"]).to(dev).float()[:, :self.n_ogrenilen]
            z = o["z"].to(dev).float()
            zr = o["z_ham"].to(dev).float()
            mse = (((z_hat - z) ** 2) * maske).sum() / maske.sum().clamp(min=1.0)
            hedef = torch.cat([(zr > esik).float(), (zr < -esik).float()], -1)
            m2 = torch.cat([maske, maske], -1)
            bce = (F.binary_cross_entropy_with_logits(pl, hedef, reduction="none") * m2).sum() / m2.sum().clamp(min=1.0)
            z_ort = z_ort + mse; bce_ort = bce_ort + bce; n += 1
        z_ort, bce_ort = z_ort / n, bce_ort / n
        return z_ort + p_w * bce_ort, {"mse": float(z_ort), "bce": float(bce_ort)}

    def uret(self, ornek, max_yeni=900, kanit_kisitla=False):
        """Tek olay icin JSON uretir. 'kanitlar' dizisi modelin olcum basliginden BIREBIR kopyalanir (ZorlaIslemci: ilgili token'lar
        zorlanir, KV onbellegine girer; LLM sonraki metni bunu gorerek yazar). Dondurur (metin, json|None, ek{sinif_p, z_tah, olcum})."""
        E, _, algi, bilgi = self.girdi_kur(ornek)
        s_lg, _, _ = self.kafalar(algi)
        with torch.no_grad():
            z_tah, _ = self.olcum_tahmin(ornek)
        M = torch.ones(1, E.shape[0], dtype=torch.long, device=E.device)
        isl = LogitsProcessorList([ZorlaIslemci(self._ids(KANIT_DIZI_LIT), bilgi["zorla_ids"]), ZorlaIslemci(self._ids(OLASI_LIT), bilgi["olasi_ids"]),
                                   KalipZorlaIslemci(self.tok, self._ids(KALIP_LIT), [[ifade for ifade, _ in ol] for ol in bilgi["olcum"]],
                                                     [bool((x or "").strip()) for x in ornek["transkript"]])])
        cikti = self.llm.generate(inputs_embeds=E[None].to(self.llm.get_input_embeddings().weight.dtype), attention_mask=M,
                                  max_new_tokens=max_yeni, do_sample=False, logits_processor=isl,
                                  eos_token_id=self.tok.convert_tokens_to_ids("<|im_end|>"),
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        metin = self.tok.decode(cikti[0], skip_special_tokens=True)
        return metin, json_ayikla(metin), {"sinif_p": F.softmax(s_lg.float(), -1).cpu(), "z_tah": z_tah.float().cpu(), "olcum": bilgi["olcum"], "olasi": bilgi["olasi"]}

    @torch.no_grad()
    def uret_sablon(self, ornek, fabrika, max_serbest=400, slot_max=6, debug=None):
        """KOD-ZORLANAN uretim: LLM yalniz parca basina 'icerik_duygusu' degerini ve 'genel_yorum' metnini YAZAR; geri kalan tum JSON KOD tarafindan hesaplanip
        (zorlanan parcalar tek forward'da, KV onbellegine 'prefill' olarak) eklenir. Sira: (a) olcumler/kanit/olasi (girdi_kur) -> (b) LLM icerik_duygusu -> (c) KOD: cozumleme/uyum/yorum/
        ruh_hali/tutum/duygu(sinif basligi)/guven/kaliplar -> (d) LLM genel_yorum. fabrika(sinif_p [C,8] numpy, bilgi) -> hedef_v2.SablonUretici. debug: {'slot': [deger..], 'etiket': ...} (test).
        Dondurur (metin, json|None, ek{sinif_p, olcum, olasi, serbest_token, zorla_token, sure_adim})."""
        E, _, algi, bilgi = self.girdi_kur(ornek)
        s_lg, _, _ = self.kafalar(algi)
        probs = F.softmax(s_lg.float(), -1).cpu().numpy()
        ut = fabrika(probs, bilgi)
        dev = E.device
        eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        out = self.llm(inputs_embeds=E[None].to(self.llm.get_input_embeddings().weight.dtype), use_cache=True)
        pkv, logits = out.past_key_values, out.logits[0, -1]
        gen, serbest, zorla = [], 0, 0

        def ileri(ids):
            nonlocal pkv
            o = self.llm(input_ids=torch.tensor([ids], device=dev), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            return o.logits[0, -1]

        ids = self._ids(ut.ilk())
        logits = ileri(ids); gen += ids; zorla += len(ids)
        son_bitti = False
        for i in range(ut.n):
            slot = []
            if debug and debug.get("slot"):                          # test: slot degerini zorla (hedefle birebir karsilastirma)
                slot = self._ids(debug["slot"][i] + '"')
                logits = ileri(slot); gen += slot
                deger = debug["slot"][i]
            else:
                for _ in range(slot_max):
                    tkn = int(torch.argmax(logits))
                    gen.append(tkn); slot.append(tkn); serbest += 1
                    logits = ileri([tkn])
                    if '"' in self.tok.decode(slot):
                        break
                deger = self.tok.decode(slot).split('"')[0].strip()
            metin, son_bitti = ut.parca_devam(i, deger)
            ids = self._ids(metin)
            logits = ileri(ids); gen += ids; zorla += len(ids)
        if not ut.bitti:                                             # genel_yorum: LLM yazar (bitis: '"}}' + <|im_end|>)
            if debug and debug.get("genel") is not None:
                ids = self._ids(debug["genel"])
                gen += ids
            else:
                for _ in range(max_serbest):
                    tkn = int(torch.argmax(logits))
                    if tkn == eos:
                        break
                    gen.append(tkn); serbest += 1
                    logits = ileri([tkn])
        metin = self.tok.decode(gen)
        j = json_ayikla(metin)
        en_olasi = (getattr(ut, "olay", None) or {}).get("en_olasi")
        if j is not None and en_olasi is not None and isinstance(j.get("olay"), dict):
            j["olay"]["en_olasi"] = en_olasi                         # teknik alan: zorlanan metinde YOK, ayristirilan JSON'a eklenir (kullaniciya gosterilen metin degismez)
        return metin, j, {"sinif_p": probs, "olcum": bilgi["olcum"], "olasi": bilgi["olasi"], "serbest_token": serbest, "zorla_token": zorla, "icerik": list(ut.degerler), "en_olasi": en_olasi}

    def analiz(self, kareler, ses, fps=25.0, dil=None, kisi_normu=None, **uret_kw):
        """HAM video + HAM ses -> JSON. kareler: [T,3,H,W] uint8 RGB; ses: 1-D float 16 kHz.
        3 sn'lik parcalara boler, her parca icin on_hesapla + Whisper decoder transkripti + Qwen uretimi."""
        c = self.config
        sure = max(len(ses) / c.ses_sr, kareler.shape[0] / fps if kareler is not None else 0.0)
        kenarlar = [(i * c.chunk_sn, min((i + 1) * c.chunk_sn, sure)) for i in range(max(1, math.ceil(sure / c.chunk_sn - 1e-6)))]
        kp, sp, ts = [], [], []
        for a, b in kenarlar:
            ts.append(f"{a:.1f}-{b:.1f}")
            sp.append(ses[int(a * c.ses_sr): int(b * c.ses_sr)] if ses is not None else None)
            if kareler is None:
                kp.append(None)
            else:
                fa, fb = int(a * fps), max(int(b * fps), int(a * fps) + 1)
                kp.append(kareler[fa:fb])
        on = self.on_hesapla(kp, sp)
        tr = self.transkript_uret(sp, dil)
        ornek = dict(goz=on["goz"], kulak=on["kulak"], goz_var=on["goz_var"], ses_var=on["ses_var"], t=ts, transkript=tr,
                     bolge=on["bolge"], geo=on["geo"], yuz_bulunan=on["yuz_bulunan"], kutu_olcek=on["kutu_olcek"])
        if kisi_normu is not None:
            ornek["norm_goz"], ornek["norm_kulak"] = kisi_normu
        metin, j, ek = self.uret(ornek, **uret_kw)
        return {"json": j, "metin": metin, "transkript": tr, "parca_t": ts, **ek}


DavranisConfig.register_for_auto_class()
DavranisModel.register_for_auto_class("AutoModel")

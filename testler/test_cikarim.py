"""Birim testleri (CPU, model agirligi/GPU gerekmez; sentetik veri). Calistir: python -m unittest discover -s testler -v"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
import cikarim_v211 as C  # noqa: E402


def _pop():
    with open(os.path.join(KOK, "yapilandirma", "referans_populasyon.json"), encoding="utf-8") as f:
        return json.load(f)


class GenelDuygu(unittest.TestCase):
    def test_icerik_oncelikli_dusuk_guven(self):
        p = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]                       # max 0,2 < tau
        slot = [{"mutluluk": np.log(0.7), "notr": np.log(0.2), "yok": np.log(0.1)}]
        g = C.genel_duygu(p, slot)
        self.assertEqual(g["kaynak"], "icerik")
        self.assertEqual(g["sinif"], "Mutluluk")
        self.assertAlmostEqual(g["dagilim"]["Mutluluk"], 0.7 / 0.9, places=3)

    def test_yuksek_guven_davranis(self):
        p = [0.05, 0.8, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
        g = C.genel_duygu(p, [{"korku": np.log(0.9), "yok": np.log(0.1)}])
        self.assertEqual((g["kaynak"], g["sinif"], g["secici"]), ("davranis", "Mutluluk", True))

    def test_icerik_yok_ise_davranis(self):
        p = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
        g = C.genel_duygu(p, [{"yok": np.log(1.0)}])
        self.assertEqual(g["kaynak"], "davranis")
        self.assertFalse(g["icerik_var"])
        self.assertFalse(g["secici"])                                       # 0,2 < 0,55

    def test_slot_hata_ve_bos(self):
        ic, w, ok = C.soft_icerik([{"hata": "x"}])
        self.assertEqual((float(ic.sum()), w, ok), (0.0, 0.0, False))
        self.assertFalse(C.soft_icerik([])[2])


class Yapilandirma(unittest.TestCase):
    def test_sha(self):
        c = C.yapilandirma()
        self.assertTrue(c["_sha256"].startswith("74b470be"))
        self.assertEqual(c["parametre"]["tau_tum"], 0.5)
        self.assertEqual(c["genel_duygu_esigi"], 0.55)
        self.assertEqual(c["baslik"], "agirlikli_us05")


class OlcumDili(unittest.TestCase):
    def test_yasak_sozcuk_yok_ve_olcum_biçimi(self):
        from davranis_model import olcum_dili as OD
        for sinyal, z in [("browDown", 2.8), ("mouthSmile", 1.4), ("jawOpen", 1.7)]:
            s = OD.olcum_cumlesi(sinyal, z, 1, "genel", "olasi")
            self.assertTrue(s.startswith("ölçümde"), s)
            self.assertIn("z=", s)
            self.assertFalse(any(w in s for w in OD.YASAK_SOZCUKLER))
        self.assertIsNone(OD.olcum_cumlesi("browDown", 0.2, 1, "genel", None))     # esik alti

    def test_kanit_icin_kisi_referansi_gerekir(self):
        from davranis_model import dogrudan_olcum as D
        pop = _pop()
        yuz = np.array(pop["yuz_ort"], float)
        std = np.array(pop["yuz_std"], float)
        isim = pop["yuz_isim"]
        for ad in ("browDownLeft", "browDownRight"):
            yuz[isim.index(ad)] += 3.5 * std[isim.index(ad)]
        ses = np.array(pop["ses_ort"], float)
        ol = D.isaretler(yuz, ses, pop, None)                                         # gecmis yok -> 'genel'
        self.assertTrue(ol and all(x["katman"] == "olasi" and x["baglam"] == "genel" for x in ol))
        g = D.KisiGecmisi()
        rng = np.random.default_rng(0)
        for _ in range(12):
            g.ekle(np.array(pop["yuz_ort"]) + rng.normal(0, 0.1, 52) * std, ses, 3.0)
        ol2 = D.isaretler(yuz, ses, pop, g)
        self.assertTrue(any(x["katman"] == "kanit" and x["sinyal"] == "browDown" for x in ol2), ol2)
        self.assertTrue(all(x["cumle"].startswith("ölçümde") for x in ol2))


class HizliCikti(unittest.TestCase):
    def test_hizli_cikti_kapali_alan_ve_yasak_yok(self):
        from davranis_model import dogrudan_olcum as D
        pop = _pop()
        yuz = np.array(pop["yuz_ort"], float)
        std = np.array(pop["yuz_std"], float)
        isim = pop["yuz_isim"]
        for ad in ("mouthSmileLeft", "mouthSmileRight", "browDownLeft", "browDownRight"):
            yuz[isim.index(ad)] += 1.8 * std[isim.index(ad)]
        ol = D.isaretler(yuz, np.array(pop["ses_ort"], float), pop, None)
        r = C.hizli_cikti(ol, "tam", 3.0)
        self.assertIn("uyari", r)
        self.assertTrue(r["olcum"])
        self.assertFalse(r.get("yorum_hatasi"), r.get("yorum_hatasi"))
        yorum = (r["yorum"] or "").lower()
        for yasak in ("yalan söylüyor", "kızgın", "kaşlarını çattı", "gülümsedi"):
            self.assertNotIn(yasak, yorum)
        self.assertNotIn("uyumlu", yorum.replace("uyumlu olmayan", ""))          # uyum iddiasi kapali
        self.assertIn("ölçümde", yorum)

    def test_isaretsiz(self):
        r = C.hizli_cikti([], "sadece_kulak", 2.0)
        self.assertEqual(r["olcum"], [])
        self.assertIsNotNone(r["yorum"])


class MedyaOku(unittest.TestCase):
    def test_wav_ve_video(self):
        import av
        import wave
        d = tempfile.mkdtemp()
        w = os.path.join(d, "a.wav")
        y = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000 * 2) / 16000) * 32767).astype(np.int16)
        with wave.open(w, "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(y.tobytes())
        kare, ses, sure = C.medya_oku(w)
        self.assertIsNone(kare)
        self.assertAlmostEqual(sure, 2.0, delta=0.1)
        self.assertEqual(ses.dtype, np.float32)
        v = os.path.join(d, "v.mp4")
        with av.open(v, "w") as c:
            s = c.add_stream("libx264", rate=10)
            s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
            for i in range(20):
                fr = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 10, np.uint8), format="rgb24")
                for p in s.encode(fr):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        kare, ses, sure = C.medya_oku(v)
        self.assertEqual(kare.shape, (8, 64, 64, 3))
        self.assertIsNone(ses)
        self.assertAlmostEqual(sure, 2.0, delta=0.3)


class OlcumKatmaniSentetik(unittest.TestCase):
    def test_yuzsuz_kare_ve_ses(self):
        ok = C.OlcumKatmani()
        kare = np.full((8, 96, 96, 3), 127, np.uint8)
        ses = (0.1 * np.random.default_rng(1).standard_normal(16000 * 2)).astype(np.float32)
        ham = ok.ham_olcum(kare, ses)
        self.assertIsNone(ham["yuz"])
        self.assertEqual(ham["bulunan_kare"], 0)
        self.assertEqual(len(ham["ses"]), 4)
        ol, baglam = ok.isaretler(ham, None)
        self.assertEqual(baglam, "genel")
        for x in ol:
            self.assertEqual(x["katman"], "olasi")


if __name__ == "__main__":
    unittest.main()

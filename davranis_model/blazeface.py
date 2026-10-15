"""BlazeFace (MediaPipe short-range yuz dedektoru) torch portu - yalnizca ileri gecis agi.

Kaynak: hollance/BlazeFace-PyTorch (Matthijs Hollemans, 2019), MediaPipe ile ayni lisans: Apache License 2.0.
Agirliklar MediaPipe'in face_detection_short_range.tflite modelinden donusturulmustur (BatchNorm katlanmis).
Burada yalnizca on_hesapla'nin ihtiyac duydugu kisim var: agi + kutu cozme. Yuz kirpma mantigi modeling_davranis.py'de.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(self.convs(h) + x)


class BlazeFace(nn.Module):
    """128x128 giris (RGB, [-1,1]); cikis: (kutu regresyonu [b,896,16], skor logit'i [b,896,1])."""

    num_anchors = 896
    num_coords = 16
    olcek = 128.0

    def __init__(self):
        super().__init__()
        self.backbone1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=False),
            BlazeBlock(24, 24), BlazeBlock(24, 28), BlazeBlock(28, 32, stride=2), BlazeBlock(32, 36),
            BlazeBlock(36, 42), BlazeBlock(42, 48, stride=2), BlazeBlock(48, 56), BlazeBlock(56, 64),
            BlazeBlock(64, 72), BlazeBlock(72, 80), BlazeBlock(80, 88),
        )
        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2), BlazeBlock(96, 96), BlazeBlock(96, 96), BlazeBlock(96, 96), BlazeBlock(96, 96),
        )
        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        # anchor'lar (896,4): [x_merkez, y_merkez, 1, 1] (normalize); agirlikla birlikte safetensors'a girer
        self.register_buffer("anchors", torch.zeros(self.num_anchors, 4), persistent=True)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]
        x = self.backbone1(x)
        h = self.backbone2(x)
        c1 = self.classifier_8(x).permute(0, 2, 3, 1).reshape(b, -1, 1)
        c2 = self.classifier_16(h).permute(0, 2, 3, 1).reshape(b, -1, 1)
        r1 = self.regressor_8(x).permute(0, 2, 3, 1).reshape(b, -1, 16)
        r2 = self.regressor_16(h).permute(0, 2, 3, 1).reshape(b, -1, 16)
        return torch.cat((r1, r2), 1), torch.cat((c1, c2), 1)

    def kutular(self, x_pm1, min_skor=0.5, iou_esik=0.3):
        """x_pm1: [b,3,128,128] ([-1,1]). Her goruntu icin en guvenilir yuz: (kutu [4] = ymin,xmin,ymax,xmax normalize,
        skor, anahtar_noktalar [6,2]) ya da (None, 0, None). Ust uste binen adaylar skor agirlikli ortalanir (BlazeFace makalesindeki karisim)."""
        raw, sk = self(x_pm1)
        a = self.anchors
        xc = raw[..., 0] / self.olcek * a[:, 2] + a[:, 0]
        yc = raw[..., 1] / self.olcek * a[:, 3] + a[:, 1]
        w = raw[..., 2] / self.olcek * a[:, 2]
        h = raw[..., 3] / self.olcek * a[:, 3]
        kutu = torch.stack([yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2], -1)        # [b,896,4]
        # 6 anahtar nokta (sag goz, sol goz, burun ucu, agiz, sag kulak, sol kulak): normalize (x,y) [b,896,6,2]
        kpx = torch.stack([raw[..., 4 + 2 * k] / self.olcek * a[:, 2] + a[:, 0] for k in range(6)], -1)
        kpy = torch.stack([raw[..., 5 + 2 * k] / self.olcek * a[:, 3] + a[:, 1] for k in range(6)], -1)
        kp_all = torch.stack([kpx, kpy], -1)
        skor = sk.clamp(-100, 100).sigmoid().squeeze(-1)                                 # [b,896]
        sonuc = []
        for i in range(x_pm1.shape[0]):
            en, j = skor[i].max(0)
            if float(en) < min_skor:
                sonuc.append((None, 0.0, None))
                continue
            k0 = kutu[i, j]
            aday = skor[i] >= min_skor
            kb, sb = kutu[i][aday], skor[i][aday]
            ymin = torch.maximum(kb[:, 0], k0[0]); xmin = torch.maximum(kb[:, 1], k0[1])
            ymax = torch.minimum(kb[:, 2], k0[2]); xmax = torch.minimum(kb[:, 3], k0[3])
            ic = (ymax - ymin).clamp(min=0) * (xmax - xmin).clamp(min=0)
            alan0 = (k0[2] - k0[0]) * (k0[3] - k0[1])
            alanb = (kb[:, 2] - kb[:, 0]) * (kb[:, 3] - kb[:, 1])
            iou = ic / (alan0 + alanb - ic + 1e-9)
            m = iou > iou_esik
            wgt = sb[m] / sb[m].sum()
            kpb = kp_all[i][aday][m]
            sonuc.append(((kb[m] * wgt[:, None]).sum(0), float(en), (kpb * wgt[:, None, None]).sum(0)))
        return sonuc

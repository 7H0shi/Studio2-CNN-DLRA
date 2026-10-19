"""Arsitektur Submission: ConvNeXt-Tiny dengan Resolusi Tinggi (288x288) & Test-Time Augmentation (TTA).

Kontrak Submission:
  - get_model() hanya MEMBANGUN arsitektur (tanpa mengunduh bobot pretrained di sini).
  - get_transform() harus SAMA dengan transform validasi saat melatih model.
  - Output model: logits [batch, 5], urutan kelas:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 288
RESIZE_SIZE = 320


class ConvNeXtWithTTA(nn.Module):
    """ConvNeXt-Tiny dengan integrasi Test-Time Augmentation (TTA) otomatis saat evaluasi."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        in_features = base.classifier[2].in_features  # 768
        base.classifier[2] = nn.Linear(in_features, num_classes)

        # Pertahankan struktur nama layer asli agar state_dict 100% kompatibel
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

    def _forward_single(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        # Saat mode training (model.train()), jalankan forward standar
        if self.training:
            return self._forward_single(x)

        # Saat mode evaluasi (model.eval()), jalankan Test-Time Augmentation (TTA):
        # Rata-ratakan logits dari citra asli dan citra hasil flip horizontal
        logits_orig = self._forward_single(x)
        logits_flip = self._forward_single(torch.flip(x, dims=[-1]))
        return 0.5 * (logits_orig + logits_flip)


def get_model():
    """Membangun arsitektur ConvNeXt-Tiny + TTA dengan linear head 5 kelas."""
    return ConvNeXtWithTTA(num_classes=NUM_CLASSES)


def get_transform():
    """Preprocessing evaluasi/validasi resolusi 288x288."""
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

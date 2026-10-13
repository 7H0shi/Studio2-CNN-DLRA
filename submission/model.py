"""Arsitektur Submission: ConvNeXt-Tiny dengan head 5 kelas sampah.

Kontrak Submission:
  - get_model() hanya MEMBANGUN arsitektur (tanpa mengunduh bobot pretrained di sini).
  - get_transform() harus SAMA dengan transform validasi saat melatih model.
  - Output model: logits [batch, 5], urutan kelas:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    """Membangun arsitektur ConvNeXt-Tiny dengan linear head 5 kelas."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    """Preprocessing evaluasi/validasi standar ConvNeXt-Tiny."""
    return transforms.Compose([
        transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

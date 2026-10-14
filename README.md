# Studio 2 — Klasifikasi Sampah · Kelompok 10 - Alcyon

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1 | Ardiansyah Fernando | 123140102 | Arsitektur Model, Evaluasi Submission, Pelatihan Model |
| 2 | Yudhistira Novianto Kurniawan | 123140138 | Preprocessing, dokumentasi |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt Tiny Pre-trained
- **Strategi pelatihan**:
  - Optimizer: AdamW
  - Scheduler: CosineAnnealingLR
  - Epoch: 10-25 epoch
  - Augmentasi: RandomHorizontalFlip, RandomResizedCrop, RandomRotation, dan ColorJitter
  - Loss: Cross Entropy Loss (dengan Label Smoothing 0.1)
- **Data tambahan**: Tidak ada
- **Ukuran input & preprocessing**: 224 * 224, normalisasi ImageNet

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | 9b1b2f9 | 0.9640 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt) | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- Baseline awal ResNet-18 (8 epoch) mencapai Macro-F1 sekitar ~0.92.
- Penggunaan arsitektur modern ConvNeXt-Tiny pre-trained dengan augmentasi citra dan scheduler Cosine Annealing berhasil meningkatkan representasi fitur secara signifikan, mencapai Macro-F1 validasi lokal **0.9640** (Akurasi: **96.83%**).

## Cara Reproduksi

```bash
pip install -r submission/requirements.txt
python train_convnext.py --epochs 10            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
train_convnext.py   # kode pelatihan model kelompok
...                 # kode/notebook pelatihan lainnya
```

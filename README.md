# Studio 2 — Klasifikasi Sampah · Kelompok 10 - Alcyon

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1 | Ardiansyah Fernando | 123140102 | Arsitektur Model, Evaluasi Submission, Pelatihan Model |
| 2 | Yudhistira Novianto Kurniawan | 123140138 | Preprocessing, DataReader, Dokumentasi |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt Tiny Pre-trained (dengan integrasi *Test-Time Augmentation* / TTA saat evaluasi)
- **Strategi pelatihan**:
  - Optimizer: AdamW (`lr=8e-5` tahap 1, `lr=2e-5` *high-resolution fine-tuning*, `weight_decay=1e-2`)
  - Scheduler: CosineAnnealingLR
  - Epoch: 10–15 epoch (ditambah 5 epoch *high-res fine-tuning*)
  - Augmentasi: RandomHorizontalFlip, RandomResizedCrop (288×288), RandomRotation (15°), ColorJitter, dan RandomErasing
  - Loss: **Class-Weighted Cross Entropy Loss (CWL)** (dengan Label Smoothing 0.08) untuk menyeimbangkan kelas minoritas (`kertas_kardus`)
  - Evaluasi: **Test-Time Augmentation (TTA)** otomatis merata-ratakan prediksi gambar asli dan *horizontal flip*
- **Data tambahan**: Tidak ada
- **Ukuran input & preprocessing**: 288 * 288 (`Resize(320)` $\rightarrow$ `CenterCrop(288)`), normalisasi ImageNet

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | c26b867 | 0.9640 | 0.9474 |
| Preview 2 (21 Okt) | f954e22 | 0.9725 | |
| Final (25 Okt) | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- **Baseline Awal**: ResNet-18 (8 epoch) mencapai Macro-F1 sekitar ~0.9159.
- **Preview 1 (ConvNeXt-Tiny 224×224)**: Mencapai Macro-F1 validasi lokal **0.9640** dan skor resmi asisten **0.9474** (Peringkat 5 dari 18 kelompok). Analisis *confusion matrix* menunjukkan kelas `kertas_kardus` menjadi titik terlemah (F1 lokal: 0.9143) karena jumlah sampel paling sedikit (311 citra latih) dan kemiripan tekstur dengan `styrofoam`.
- **Preview 2 (Upgrade Resolusi 288×288 + CWL + TTA)**:
  1. Menaikkan resolusi input ke **288×288** agar tekstur serat kertas/kardus vs pori-pori styrofoam tertangkap lebih tajam.
  2. Menerapkan **Class-Weighted Loss (CWL)** berbanding terbalik dengan frekuensi kelas (`kertas_kardus` diberi bobot 1.51×).
  3. Memasang **Test-Time Augmentation (TTA)** otomatis saat mode `.eval()`.
  4. Hasilnya: F1 kelas `kertas_kardus` melonjak dari **0.9143 $\rightarrow$ 0.9423**, `styrofoam` naik dari **0.9608 $\rightarrow$ 0.9706**, dan skor **Macro-F1 validasi lokal meningkat signifikan menjadi 0.9725** (Akurasi: **97.56%**).

## Cara Reproduksi

```bash
pip install -r submission/requirements.txt
python train_convnext.py --epochs 12                       # melatih dari pretrained ImageNet
python train_convnext.py --finetune --epochs 5 --lr 2e-5   # high-res fine-tuning dengan CWL
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() (ConvNeXt-Tiny + TTA) dan get_transform() (288x288)
  model.pth         # state_dict model terbaik (~53 MB FP16)
  requirements.txt  # daftar paket dependensi
datareader.py       # modul custom PyTorch Dataset (TrashDataset)
evaluate.py         # salinan skrip evaluasi resmi
train_convnext.py   # kode pelatihan model kelompok (CWL + High-Res + AMP)
train_baseline.py   # kode baseline awal
```

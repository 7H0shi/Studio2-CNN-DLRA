"""Skrip Pelatihan ConvNeXt-Tiny (Upgrade Preview 2) — Studio 2 (IF25-40401).

Peningkatan (Upgrade) Preview 2:
  1. Resolusi Input Tinggi (288x288) -> Menangkap detail tekstur kertas/kardus vs styrofoam.
  2. Class-Weighted Loss (CWL) -> Mengatasi class imbalance (memberi bobot lebih pada kelas minoritas).
  3. Test-Time Augmentation (TTA) -> Otomatis aktif saat model.eval() melalui ConvNeXtWithTTA.
  4. Penyimpanan Bobot FP16 -> Ukuran model.pth ~53 MB (aman di bawah batas 100 MB GitHub).
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datareader import TrashDataset
from submission.model import IMG_SIZE, NUM_CLASSES, get_model, get_transform

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]


def set_seed(seed=42):
    """Mengunci keacakan untuk Windows (CUDA), Mac (MPS), dan CPU."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def calculate_metrics(y_true, y_pred, k=NUM_CLASSES):
    """Menghitung Macro-F1 dan Akurasi sesuai dengan skrip resmi evaluate.py."""
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    f1s = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k)) - tp
        fn = sum(cm[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(cm[c][c] for c in range(k)) / max(1, len(y_true))
    macro_f1 = sum(f1s) / k
    return macro_f1, acc, dict(zip(CLASSES, f1s))


def make_local_split(src, dst, frac=0.15, seed=42):
    """Membuat folder my_train/ dan my_val/ jika belum ada."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists() and any(dst_va.iterdir()):
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def compute_class_weights(dataset, num_classes, device):
    """Menghitung bobot kelas berbanding terbalik dengan frekuensi kelas (Class-Weighted Loss)."""
    counts = [0] * num_classes
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    total = sum(counts)
    # Rumus Inverse Class Frequency: total / (num_classes * count_c)
    weights = [total / (num_classes * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def save_fp16_checkpoint(model, path="submission/model.pth"):
    """Menyimpan state_dict dalam presisi FP16 (~53 MB) agar aman di-push ke GitHub (<100 MB)."""
    state = model.state_dict()
    state_fp16 = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    torch.save(state_fp16, path)


@torch.no_grad()
def evaluate_validation(model, va_dl, device):
    """Evaluasi model pada validation set (otomatis mengaktifkan TTA karena model.eval())."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in va_dl:
        images = images.to(device)
        outputs = model(images)  # TTA aktif di sini
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return calculate_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Latih ConvNeXt-Tiny Upgrade (288px + CWL + TTA)")
    parser.add_argument("--data", default="./train", help="Path ke folder dataset train")
    parser.add_argument("--epochs", type=int, default=12, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=24, help="Ukuran batch (24 optimal untuk 288x288 di VRAM 6GB)")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate awal")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay untuk AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--finetune", action="store_true", help="Lanjutkan fine-tuning resolusi tinggi dari submission/model.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"--> Menggunakan Device : {device.upper()}")
    if device == "cuda":
        print(f"--> Nama GPU           : {torch.cuda.get_device_name(0)}")
    print(f"--> Resolusi Input     : {IMG_SIZE}x{IMG_SIZE} piksel (TTA Aktif saat Evaluasi)")

    tr_dir, va_dir = make_local_split(args.data, ".", frac=0.15, seed=args.seed)

    # Augmentasi data training pada resolusi 288x288
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    val_tf = get_transform()

    tr_ds = TrashDataset(tr_dir, train_tf)
    va_ds = TrashDataset(va_dir, val_tf)
    print(f"--> Dataset            : Train = {len(tr_ds)} citra | Val = {len(va_ds)} citra")

    # Hitung Class-Weighted Loss (CWL)
    class_weights, counts = compute_class_weights(tr_ds, NUM_CLASSES, device)
    weight_info = " | ".join([f"{c}: {w:.2f} (n={n})" for c, w, n in zip(CLASSES, class_weights.tolist(), counts)])
    print(f"--> Bobot CWL          : {weight_info}")

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    # Inisialisasi arsitektur dari submission/model.py
    model = get_model()
    Path("submission").mkdir(parents=True, exist_ok=True)
    sub_pth = Path("submission/model.pth")

    if args.finetune and sub_pth.exists():
        print("--> Fine-tuning resolusi tinggi dari bobot submission/model.pth...")
        model.load_state_dict(torch.load(sub_pth, map_location="cpu", weights_only=True))
    else:
        print("--> Memuat pretrained weights ImageNet-1K ke ConvNeXt-Tiny...")
        pretrained = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        pretrained_state = {k: v for k, v in pretrained.state_dict().items() if not k.startswith("classifier.2.")}
        model.load_state_dict(pretrained_state, strict=False)
    model = model.to(device)

    # Cek skor model.pth yang sudah ada sebelumnya agar tidak pernah menimpa dengan yang lebih buruk
    best_macro_f1 = 0.0
    if sub_pth.exists():
        try:
            existing_model = get_model()
            existing_model.load_state_dict(torch.load(sub_pth, map_location="cpu", weights_only=True))
            existing_model.to(device)
            init_f1, init_acc, _ = evaluate_validation(existing_model, va_dl, device)
            best_macro_f1 = init_f1
            print(f"--> Baseline model.pth saat ini (288px + TTA): Val Macro-F1 = {init_f1:.4f} (Acc: {init_acc:.4f})")
            del existing_model
        except Exception as e:
            print(f"--> Info: memulai rekor baru ({e})")

    # Optimizer AdamW + Cosine Annealing + Class-Weighted CrossEntropyLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("\n========================= MULAI PELATIHAN =========================")
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in tr_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(tr_ds)

        # Evaluasi pada data validasi (dengan TTA)
        macro_f1, acc, per_cls_f1 = evaluate_validation(model, va_dl, device)
        elapsed = time.time() - t_start

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f} | Val Macro-F1 (TTA): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_fp16_checkpoint(model, "submission/model.pth")
            print(f"  --> [REKOR BARU TERSIMPAN] Skor Macro-F1: {best_macro_f1:.4f} -> submission/model.pth")
            f1_summary = " | ".join([f"{c}: {f:.3f}" for c, f in per_cls_f1.items()])
            print(f"      F1 per kelas: {f1_summary}")

    print("\n========================= SELESAI =========================")
    print(f"Pelatihan selesai! Skor Macro-F1 Terbaik: {best_macro_f1:.4f}")
    print("Model disimpan di: submission/model.pth")
    print("Jalankan evaluasi resmi dengan:")
    print("  python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()

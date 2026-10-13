import argparse
import os
import random
import shutil
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from submission.model import NUM_CLASSES, get_model, get_transform

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]

def set_seed(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def main():
    parser = argparse.ArgumentParser(description="Latih ConvNeXt-Tiny Pretrained")
    parser.add_argument("--data", default="./train", help="Path ke folder dataset train")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate awal")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay untuk AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Menggunakan Device: {device.upper()}")
    if device == "cuda":
        print(f"--> Nama GPU          : {torch.cuda.get_device_name(0)}")

    tr_dir, va_dir = make_local_split(args.data, ".", frac=0.15, seed=args.seed)

    # Augmentasi data training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = get_transform()

    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    va_ds = datasets.ImageFolder(va_dir, val_tf)
    print(f"--> Dataset: Train = {len(tr_ds)} citra | Val = {len(va_ds)} citra | Kelas = {len(tr_ds.classes)}")

    # Gunakan num_workers=2 untuk Windows DataLoader yang cepat
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    # Inisialisasi model dari bobot pretrained ConvNeXt-Tiny resmi
    print("--> Mengunduh & memuat pretrained weights ConvNeXt-Tiny...")
    pretrained_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = pretrained_model.classifier[2].in_features
    pretrained_model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    model = pretrained_model.to(device)

    # Optimizer AdamW + Cosine Annealing Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_macro_f1 = 0.0
    Path("submission").mkdir(parents=True, exist_ok=True)

    print("\n========================= MULAI PELATIHAN =========================")
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in tr_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(tr_ds)

        # Evaluasi pada data validasi
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in va_dl:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        macro_f1, acc, per_cls_f1 = calculate_metrics(y_true, y_pred)
        elapsed = time.time() - t_start

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f}")

        # Simpan jika skor Macro-F1 membaik
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "submission/model.pth")
            print(f"  --> [BEST MODEL TERSIMPAN] Skor Macro-F1 baru: {best_macro_f1:.4f} -> submission/model.pth")
            f1_summary = " | ".join([f"{c}: {f:.3f}" for c, f in per_cls_f1.items()])
            print(f"      F1 per kelas: {f1_summary}")

    print("\n========================= SELESAI =========================")
    print(f"Pelatihan selesai! Skor Macro-F1 Terbaik: {best_macro_f1:.4f}")
    print("Model disimpan di: submission/model.pth")
    print("Jalankan evaluasi resmi dengan:")
    print("  python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()

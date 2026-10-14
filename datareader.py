from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

# Urutan kelas alfabetis resmi Studio 2
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class TrashDataset(Dataset):
    """Custom Dataset PyTorch untuk memuat dataset klasifikasi sampah 5 kelas."""

    def __init__(self, root_dir, transform=None):
        """
        Inisialisasi DataReader.
        
        Args:
            root_dir (str atau Path): Path ke folder data (misal: my_train atau my_val).
            transform (callable, optional): Transformasi / augmentasi gambar torchvision.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        # Pindai gambar untuk setiap kelas
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in EXTS:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        """Mengembalikan total jumlah gambar yang ada di dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Mengambil satu sampel gambar dan label berdasarkan indeks.
        
        Args:
            idx (int): Indeks gambar.
            
        Returns:
            tuple: (image_tensor, label_idx)
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Uji coba mandiri DataReader
    print("Menguji TrashDataset...")
    test_ds = TrashDataset("./my_val")
    print(f"Total sampel terbaca: {len(test_ds)}")
    if len(test_ds) > 0:
        sample_img, sample_lbl = test_ds[0]
        print(f"Sampel pertama: ukuran={sample_img.size}, label={CLASSES[sample_lbl]} ({sample_lbl})")
        print("TrashDataset siap digunakan!")

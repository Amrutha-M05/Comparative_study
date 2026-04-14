from torch.utils.data import DataLoader
from src.dataset import LAGDataset, collect_split_samples
from torchvision import transforms


def get_loaders(root="LAG", batch_size=16):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("Loading dataset samples...")

    train_samples = collect_split_samples(root, "train")
    test_samples = collect_split_samples(root, "test")

    print("Train samples:", len(train_samples))
    print("Test samples:", len(test_samples))

    train_dataset = LAGDataset(train_samples, transform=transform)
    test_dataset = LAGDataset(test_samples, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,              # 🔥 safer than 4 on Windows
        pin_memory=True,
        persistent_workers=True,    # 🔥 keeps workers alive (faster)
        prefetch_factor=2           # 🔥 speeds up loading
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, test_loader
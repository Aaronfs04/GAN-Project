import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.utils import set_seed

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def get_dataloaders(data_dir='./data', batch_size=128, seed=42):
    set_seed(seed)
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_raw = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_data = ConcatDataset([train_raw, test_raw])
    total = len(full_data)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full_data, [train_size, val_size, test_size], generator=gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Train batch shape:", images.shape, "Labels shape:", labels.shape)

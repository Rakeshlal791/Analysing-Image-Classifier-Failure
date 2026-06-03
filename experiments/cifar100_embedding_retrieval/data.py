from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from cifar100_meta import FINE_TO_COARSE_ID


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class IndexedCIFAR100(Dataset):
    def __init__(self, dataset: datasets.CIFAR100):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        fine = int(target)
        coarse = FINE_TO_COARSE_ID[fine]
        return {
            "image": image,
            "fine": fine,
            "coarse": coarse,
            "index": index,
        }


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def image_transform():
    return transforms.Compose([transforms.ToTensor()])


def make_cifar100(root: str | Path, train: bool, transform, download: bool = True):
    return datasets.CIFAR100(
        root=str(root),
        train=train,
        transform=transform,
        download=download,
    )


def stratified_support_split(targets, support_fraction: float, seed: int):
    by_class = {}
    for index, label in enumerate(targets):
        by_class.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    train_indices = []
    support_indices = []

    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        support_count = max(1, round(len(indices) * support_fraction))
        support_indices.extend(indices[:support_count])
        train_indices.extend(indices[support_count:])

    rng.shuffle(train_indices)
    rng.shuffle(support_indices)
    return train_indices, support_indices


def save_split(path: str | Path, train_indices, support_indices, seed: int, support_fraction: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "support_fraction": support_fraction,
        "train_indices": train_indices,
        "support_indices": support_indices,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path):
    return json.loads(Path(path).read_text())


def make_train_support_datasets(data_root, split_path, support_fraction, seed, download=True):
    base_for_split = make_cifar100(data_root, train=True, transform=eval_transform(), download=download)
    split_path = Path(split_path)
    if split_path.exists():
        split = load_split(split_path)
        train_indices = split["train_indices"]
        support_indices = split["support_indices"]
    else:
        train_indices, support_indices = stratified_support_split(
            base_for_split.targets,
            support_fraction=support_fraction,
            seed=seed,
        )
        save_split(split_path, train_indices, support_indices, seed, support_fraction)

    train_base = make_cifar100(data_root, train=True, transform=train_transform(), download=download)
    support_base = make_cifar100(data_root, train=True, transform=eval_transform(), download=download)
    return (
        Subset(IndexedCIFAR100(train_base), train_indices),
        Subset(IndexedCIFAR100(support_base), support_indices),
        base_for_split.classes,
    )


def make_support_image_dataset(data_root, split_path, support_fraction, seed, download=True):
    base_for_split = make_cifar100(data_root, train=True, transform=eval_transform(), download=download)
    split_path = Path(split_path)
    if split_path.exists():
        split = load_split(split_path)
        support_indices = split["support_indices"]
    else:
        train_indices, support_indices = stratified_support_split(
            base_for_split.targets,
            support_fraction=support_fraction,
            seed=seed,
        )
        save_split(split_path, train_indices, support_indices, seed, support_fraction)

    image_base = make_cifar100(data_root, train=True, transform=image_transform(), download=download)
    return Subset(IndexedCIFAR100(image_base), support_indices), image_base.classes


def make_train_image_dataset(data_root, download=True):
    base = make_cifar100(data_root, train=True, transform=image_transform(), download=download)
    return IndexedCIFAR100(base), base.classes


def make_test_dataset(data_root, download=True):
    base = make_cifar100(data_root, train=False, transform=eval_transform(), download=download)
    return IndexedCIFAR100(base), base.classes


def make_test_image_dataset(data_root, download=True):
    base = make_cifar100(data_root, train=False, transform=image_transform(), download=download)
    return IndexedCIFAR100(base), base.classes

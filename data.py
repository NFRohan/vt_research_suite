from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .configs import DatasetConfig

try:
    import tonic
except ImportError:
    tonic = None


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    num_classes: int
    in_channels: int
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    is_event: bool = False


@dataclass(frozen=True)
class DataSplitBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    calib_loader: DataLoader
    test_loader: DataLoader
    metadata: DatasetMetadata


def _loader_kwargs(
    loader: DataLoader,
    *,
    dataset: Dataset,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> dict:
    target_batch_size = loader.batch_size if batch_size is None else batch_size
    target_shuffle = False if shuffle is None else shuffle
    target_num_workers = loader.num_workers if num_workers is None else num_workers
    target_pin_memory = loader.pin_memory if pin_memory is None else pin_memory

    kwargs = {
        "dataset": dataset,
        "batch_size": target_batch_size,
        "shuffle": target_shuffle,
        "num_workers": target_num_workers,
        "pin_memory": target_pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "worker_init_fn": loader.worker_init_fn,
        "generator": loader.generator,
        "persistent_workers": loader.persistent_workers if target_num_workers > 0 else False,
    }

    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if target_num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    timeout = getattr(loader, "timeout", 0)
    if timeout:
        kwargs["timeout"] = timeout

    multiprocessing_context = getattr(loader, "multiprocessing_context", None)
    if multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context

    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return kwargs


def _standard_loader_runtime_kwargs(num_workers: int) -> dict:
    kwargs = {
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return kwargs


def rebuild_loader(
    loader: DataLoader,
    batch_size: int | None = None,
    *,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    return DataLoader(
        **_loader_kwargs(
            loader,
            dataset=loader.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )


def _extract_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_targets = _extract_targets(dataset.dataset)
        return [int(parent_targets[index]) for index in dataset.indices]

    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if isinstance(values, torch.Tensor):
                return [int(value) for value in values.tolist()]
            return [int(value) for value in list(values)]

    raise ValueError(
        f"Could not extract class targets from dataset type {type(dataset).__name__}. "
        "Balanced attack subsets currently require datasets exposing `targets` or `labels`."
    )


def build_subset_loader(
    loader: DataLoader,
    subset_size: int,
    seed: int,
    *,
    balanced: bool = False,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")

    base_dataset = loader.dataset
    if subset_size >= len(base_dataset):
        return rebuild_loader(loader, batch_size=batch_size, shuffle=shuffle)

    generator = torch.Generator().manual_seed(seed)

    if balanced:
        targets = _extract_targets(base_dataset)
        class_values = sorted(set(targets))
        if not class_values:
            raise ValueError("Balanced subset construction requires at least one class label.")
        per_class = subset_size // len(class_values)
        remainder = subset_size % len(class_values)
        if per_class <= 0:
            raise ValueError(
                f"subset_size={subset_size} is too small for {len(class_values)} classes."
            )

        selected_indices: list[int] = []
        for class_rank, class_value in enumerate(class_values):
            class_indices = [idx for idx, target in enumerate(targets) if target == class_value]
            target_count = per_class + (1 if class_rank < remainder else 0)
            if len(class_indices) < target_count:
                raise ValueError(
                    f"Class {class_value} only has {len(class_indices)} samples, but {target_count} were requested."
                )
            order = torch.randperm(len(class_indices), generator=generator).tolist()
            selected_indices.extend(class_indices[idx] for idx in order[:target_count])
        shuffle_order = torch.randperm(len(selected_indices), generator=generator).tolist()
        selected_indices = [selected_indices[idx] for idx in shuffle_order]
        subset_dataset = Subset(base_dataset, selected_indices)
    else:
        subset_dataset = _maybe_subset(base_dataset, subset_size, seed)

    return DataLoader(
        **_loader_kwargs(
            loader,
            dataset=subset_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    )


class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.size(1), image.size(2)
        mask = np.ones((height, width), dtype=np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)
            y1 = np.clip(y - self.length // 2, 0, height)
            y2 = np.clip(y + self.length // 2, 0, height)
            x1 = np.clip(x - self.length // 2, 0, width)
            x2 = np.clip(x + self.length // 2, 0, width)
            mask[y1:y2, x1:x2] = 0.0
        return image * torch.from_numpy(mask).to(image.dtype).expand_as(image)


class EventFramesToTensor:
    def __init__(self, resize: int = 32):
        self.resize = resize

    def __call__(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        tensor = F.interpolate(
            tensor,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
        )
        scale = tensor.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        return tensor / scale


_DATASET_REGISTRY: dict[str, DatasetMetadata] = {
    "cifar10": DatasetMetadata(
        name="cifar10",
        num_classes=10,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    ),
    "cifar100": DatasetMetadata(
        name="cifar100",
        num_classes=100,
        in_channels=3,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    "mnist_dvs": DatasetMetadata("mnist_dvs", 10, 2, is_event=True),
    "dvs128_gesture": DatasetMetadata("dvs128_gesture", 11, 2, is_event=True),
}


def get_dataset_metadata(name: str) -> DatasetMetadata:
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {name}")
    return _DATASET_REGISTRY[key]


def _maybe_subset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def _split_indices(length: int, val_fraction: float, calibration_fraction: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    if val_fraction < 0 or calibration_fraction < 0 or val_fraction + calibration_fraction >= 1.0:
        raise ValueError("val_fraction and calibration_fraction must be non-negative and sum to less than 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_count = int(length * val_fraction)
    calib_count = int(length * calibration_fraction)
    train_count = length - val_count - calib_count
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    calib_indices = indices[train_count + val_count:]
    return train_indices, val_indices, calib_indices


def _build_cifar_datasets(config: DatasetConfig) -> tuple[Dataset, Dataset]:
    if config.augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
            ]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    if config.name == "cifar10":
        train_dataset = datasets.CIFAR10(config.root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(config.root, train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR100(config.root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(config.root, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def _build_cifar_split_datasets(config: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    if config.augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
            ]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
    eval_transform = transforms.Compose([transforms.ToTensor()])

    dataset_cls = datasets.CIFAR10 if config.name == "cifar10" else datasets.CIFAR100
    train_aug_dataset = dataset_cls(config.root, train=True, download=True, transform=train_transform)
    train_eval_dataset = dataset_cls(config.root, train=True, download=True, transform=eval_transform)
    test_dataset = dataset_cls(config.root, train=False, download=True, transform=eval_transform)
    return train_aug_dataset, train_eval_dataset, test_dataset


def _build_event_datasets(config: DatasetConfig) -> tuple[Dataset, Dataset]:
    if tonic is None:
        raise ImportError(
            "Event datasets require `tonic`. Install it before running MNIST-DVS or DVS128 Gesture experiments."
        )

    if config.name == "mnist_dvs":
        dataset_cls = tonic.datasets.MNISTDVS
    else:
        dataset_cls = tonic.datasets.DVSGesture

    frame_transform = tonic.transforms.Compose(
        [
            tonic.transforms.ToFrame(sensor_size=dataset_cls.sensor_size, n_time_bins=config.event_bins),
            EventFramesToTensor(resize=config.event_resize),
        ]
    )
    train_dataset = dataset_cls(save_to=config.root, train=True, transform=frame_transform)
    test_dataset = dataset_cls(save_to=config.root, train=False, transform=frame_transform)
    return train_dataset, test_dataset


def build_dataloaders(config: DatasetConfig) -> tuple[DataLoader, DataLoader, DatasetMetadata]:
    metadata = get_dataset_metadata(config.name)
    if metadata.is_event:
        train_dataset, test_dataset = _build_event_datasets(config)
    else:
        train_dataset, test_dataset = _build_cifar_datasets(config)

    train_dataset = _maybe_subset(train_dataset, config.train_subset, config.seed)
    test_dataset = _maybe_subset(test_dataset, config.test_subset, config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=not metadata.is_event,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )
    return train_loader, test_loader, metadata


def build_data_splits(config: DatasetConfig) -> DataSplitBundle:
    metadata = get_dataset_metadata(config.name)

    if metadata.is_event:
        train_dataset, test_dataset = _build_event_datasets(config)
        train_eval_dataset = train_dataset
    else:
        train_dataset, train_eval_dataset, test_dataset = _build_cifar_split_datasets(config)

    train_indices, val_indices, calib_indices = _split_indices(
        len(train_dataset),
        config.val_fraction,
        config.calibration_fraction,
        config.seed,
    )

    val_subset_size = config.val_subset if config.val_subset is not None else config.test_subset
    calib_subset_size = config.calib_subset if config.calib_subset is not None else config.test_subset

    train_subset = _maybe_subset(Subset(train_dataset, train_indices), config.train_subset, config.seed)
    val_subset = _maybe_subset(Subset(train_eval_dataset, val_indices), val_subset_size, config.seed)
    calib_subset = _maybe_subset(Subset(train_eval_dataset, calib_indices), calib_subset_size, config.seed)
    test_subset = _maybe_subset(test_dataset, config.test_subset, config.seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=not metadata.is_event,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )
    calib_loader = DataLoader(
        calib_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        **_standard_loader_runtime_kwargs(config.num_workers),
    )

    return DataSplitBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        calib_loader=calib_loader,
        test_loader=test_loader,
        metadata=metadata,
    )

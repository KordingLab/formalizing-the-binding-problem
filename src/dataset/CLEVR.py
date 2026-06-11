import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .ColorShape import BindingDataset


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _pil_to_chw_float(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class CLEVRDataset(BindingDataset):
    """Reader for the original pre-rendered CLEVR-style 6-object dataset."""

    def __init__(self, cfg, seed=0, num_samples=None, transform=None, output_dir=None, split="train"):
        del output_dir

        colors = list(cfg.dataset.colors)
        shapes = list(cfg.dataset.shapes)
        materials = list(cfg.dataset.materials)
        sizes = list(cfg.dataset.sizes)

        super().__init__(
            feature_groups=[colors, shapes, materials, sizes],
            distribution_mode=cfg.dataset.get("distribution_mode", "natural"),
            max_objects=cfg.dataset.get("max_objects", None),
            transform=transform,
        )

        self.cfg = cfg
        self.seed = seed if seed is not None else 0
        self.split = str(split)
        self.data_dir = str(cfg.dataset.data_dir)
        self.image_size = int(cfg.dataset.get("image_size", 224))
        self.num_total_samples = int(cfg.dataset.get("num_total_samples", 50000))
        self.second_root_start = int(cfg.dataset.get("second_root_start", self.num_total_samples // 2))

        requested = int(num_samples) if num_samples is not None else int(cfg.dataset.get("num_samples", self.num_total_samples))
        start = self._split_start()
        stop = min(start + requested, self.num_total_samples)
        if stop <= start:
            raise ValueError(
                f"No CLEVR samples available for split={self.split!r}: "
                f"start={start}, requested={requested}, total={self.num_total_samples}."
            )
        self.image_ids = np.arange(start, stop, dtype=np.int64)
        if len(self.image_ids) < requested:
            print(
                f"[warn] CLEVR split {self.split!r} requested {requested} samples, "
                f"but only {len(self.image_ids)} are available before num_total_samples={self.num_total_samples}."
            )

        self.generate_dataset()

    def _split_start(self) -> int:
        train_total = int(self.cfg.dataset.get("num_train_samples", 0))
        train_count = int(train_total * 0.8)
        val_count = int(train_total * 0.2)
        if self.split == "val":
            return train_count
        if self.split == "test":
            return train_count + val_count
        return 0

    def __len__(self):
        return len(self.image_ids)

    def _legacy_root(self, sample_id: int) -> str:
        if int(sample_id) >= self.second_root_start:
            return self.data_dir.rstrip("/") + "2"
        return self.data_dir

    def _sample_path(self, sample_id: int, kind: str) -> str:
        filename = f"CLEVR_6obj_{int(sample_id):06d}"
        subdir = "scenes_6obj" if kind == "scene" else "images_6obj"
        ext = "json" if kind == "scene" else "png"
        candidates = [
            os.path.join(self._legacy_root(sample_id), subdir, f"{filename}.{ext}"),
            os.path.join(self.data_dir, subdir, f"{filename}.{ext}"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def process_metadata(self, sample_id: int):
        metadata = _read_json(self._sample_path(sample_id, "scene"))
        objs = metadata["objects"]
        y = torch.zeros(self.group_sizes, dtype=torch.float32)
        for obj in objs:
            color_idx = self.feature_groups[0].index(obj["color"])
            shape_idx = self.feature_groups[1].index(obj["shape"])
            material_idx = self.feature_groups[2].index(obj["material"])
            size_idx = self.feature_groups[3].index(obj["size"])
            y[color_idx, shape_idx, material_idx, size_idx] = 1.0
        return y.view(-1)

    def generate_dataset(self):
        Ys, Fs, Hs = [], [], []
        for sample_id in tqdm(self.image_ids, desc=f"Loading CLEVR {self.split}", unit="sample"):
            y = self.process_metadata(int(sample_id))
            f = self.Y_to_F(y)
            h = self.get_H_Y_F(f)
            Ys.append(y)
            Fs.append(f)
            Hs.append(h)

        self.Y = torch.stack(Ys, 0).contiguous()
        self.F = torch.stack(Fs, 0).contiguous()
        self.H_Y_given_F = torch.stack(Hs, 0).contiguous()
        self.mean_H_Y_given_F = self.H_Y_given_F.mean().item()

    def __getitem__(self, idx):
        sample_id = int(self.image_ids[int(idx)])
        image = Image.open(self._sample_path(sample_id, "image")).convert("RGB")
        img = _pil_to_chw_float(image)
        if getattr(self, "transform", None) is not None:
            img = self.transform(img)
        return img, self.Y[idx], self.F[idx], self.H_Y_given_F[idx]

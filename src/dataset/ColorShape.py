import json
import math
import os
import random
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

import itertools

def _nCk(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def _log2_nCk(n: int, k: int) -> float:
    """log2(C(n,k)) with safe handling."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)

def count_no_empty_groups(a_tuple, k: int) -> int:
    """
    N_k(a_1,...,a_G): number of size-k configurations (choose k conjunctions from the
    full product space) such that every detected feature in every group appears at least once.

    Inclusion–exclusion (your Eq. ie-general-G):
      N_k(a_1,...,a_G)
      = sum_{i_1=0..a_1} ... sum_{i_G=0..a_G}
          (-1)^{sum i_g} * prod_g C(a_g, i_g) * C(prod_g (a_g - i_g), k)
    """
    

    a_tuple = tuple(int(x) for x in a_tuple)

    if k < 0 or any(x < 0 for x in a_tuple):
        return 0
    if any(x == 0 for x in a_tuple):
        # Only valid if all groups are empty and k==0 (empty configuration)
        return 1 if (all(x == 0 for x in a_tuple) and k == 0) else 0

    # Necessary feasibility bounds
    if k < max(a_tuple):
        return 0
    prod_a = 1
    for x in a_tuple:
        prod_a *= x
    if k > prod_a:
        return 0

    total = 0
    ranges = [range(ag + 1) for ag in a_tuple]  # i_g = 0..a_g
    for i_tuple in itertools.product(*ranges):
        # sign = (-1)^{sum i_g}
        sign = -1 if (sum(i_tuple) & 1) else 1

        coeff = 1
        cells = 1
        for ag, ig in zip(a_tuple, i_tuple):
            coeff *= _nCk(ag, ig)
            cells *= (ag - ig)

        ways = _nCk(cells, k) if cells >= k else 0
        total += sign * coeff * ways

    return int(max(total, 0))

class BindingDataset(Dataset):
    """
    Generic binding dataset for K feature-groups.
    Example for color-shape:
      feature_groups = [["red","blue"], ["circle","square"]]
    Conjunctions are the cartesian product across groups.
    Y: (n_conjunctions,) binary vector of which conjunctions are present.
    F: (n_features,) binary vector of which features are detected (at least once).
    H_Y_given_F: scalar, entropy H(Y|F) in bits under uniform-over
    """

    def __init__(self, feature_groups, distribution_mode="uniform", max_objects=None, transform=None):
        super().__init__()

        # feature_groups: list of lists, e.g. [[red,blue],[circle,square]]
        self.feature_groups = [list(g) for g in feature_groups]
        self.group_sizes = [len(g) for g in self.feature_groups]
        self.n_groups = len(self.feature_groups)

        # flatten feature names + indices
        self.feature_names = [name for g in self.feature_groups for name in g]
        self.n_features = len(self.feature_names)

        # group offsets into flattened feature vector
        self.group_offsets = []
        off = 0
        for sz in self.group_sizes:
            self.group_offsets.append(off)
            off += sz

        # build conjunctions = cartesian product over groups (as indices per group)
        self.conjunction_group_indices = self._cartesian_indices(self.group_sizes)
        self.n_conjunctions = len(self.conjunction_group_indices)

        # n_conjunctions x n_features binary matrix
        self.conjunction_to_feature = torch.zeros(self.n_conjunctions, self.n_features, dtype=torch.long)
        for k, idxs in enumerate(self.conjunction_group_indices):
            for gi, vi in enumerate(idxs):
                self.conjunction_to_feature[k, self.group_offsets[gi] + vi] = 1

        # sampling mode
        self.distribution_mode = distribution_mode  # "uniform"

        # storage (filled by generate_dataset in subclass)
        self.images = None
        self.Y = None
        self.F = None
        self.H_Y_given_F = None
        if max_objects is not None:
            self.max_objects = max_objects
        else: 
            self.max_objects = self.n_conjunctions

        # Optional image transform applied in subclasses' __getitem__.
        # Expected signature: transform(torch.FloatTensor[C,H,W]) -> torch.FloatTensor[C,H,W]
        self.transform = transform

    def _cartesian_indices(self, sizes):
        # returns list of tuples, each tuple has len(sizes) entries
        out = [()]
        for sz in sizes:
            out = [p + (i,) for p in out for i in range(sz)]
        return out

    def __len__(self):
        return 0 if self.Y is None else int(self.Y.shape[0])

    def __getitem__(self, idx):
        img = self._render(self.Y[idx], self.boxes_list[idx])
        return img, self.Y[idx], self.F[idx], self.H_Y_given_F[idx]

    def Y_to_F(self, Y):
        if Y.ndim == 1:
            Y2 = Y[None, :]
            squeeze = True
        else:
            Y2 = Y
            squeeze = False

        F = (Y2.to(torch.long) @ self.conjunction_to_feature) > 0
        F = F.to(torch.long)
        return F[0] if squeeze else F


    def _count_Y_consistent_with_F(self, F: torch.Tensor) -> int:
        """
        Count |Omega(F)| for general n_groups under your dataset's
        "uniform over all Y with |Y|<=max_objects" distribution.

        Let a_g be the number of detected values in group g implied by F.
        For fixed k, the number of size-k configurations that cover every detected
        feature at least once is N_k(a_1,...,a_G) (inclusion–exclusion).
        Then:
            |Omega(F)| = sum_{k=k_min..k_max} N_k(a_1,...,a_G),
        where k_min = max_g a_g, and k_max = min(prod_g a_g, max_objects).

        Special-cases:
        - all a_g == 0  -> 1 (empty configuration)
        - some a_g == 0 -> 0
        """
        F = F.to(torch.long).view(-1)

        # detected counts per group: a_g
        a = []
        for g in range(self.n_groups):
            off = self.group_offsets[g]
            sz = self.group_sizes[g]
            a.append(int(F[off : off + sz].sum().item()))
        a = tuple(a)

        if all(x == 0 for x in a):
            return 1
        if any(x == 0 for x in a):
            return 0

        k_min = max(a)

        prod_a = 1
        for x in a:
            prod_a *= x
        k_max = min(prod_a, int(self.max_objects))

        total = 0
        for k in range(k_min, k_max + 1):
            total += count_no_empty_groups(a, k)  # generalized inclusion–exclusion

        return int(max(total, 0))


    def get_H_Y_F(self, F: torch.Tensor) -> torch.Tensor:
        """
        H(Y|F) in bits for your dataset's uniform-over-Y-with-|Y|<=max_objects distribution.
        Vectorized over batch of F.
        """
        if F.ndim == 1:
            F2, squeeze = F[None, :], True
        else:
            F2, squeeze = F, False

        hs = []
        for i in range(F2.shape[0]):
            cnt = self._count_Y_consistent_with_F(F2[i])
            hs.append(math.log2(cnt) if cnt > 0 else 0.0)

        out = torch.tensor(hs, dtype=torch.float32)
        return out[0] if squeeze else out

    def get_H_Y_F_given_num_objects(self, k: int) -> float:
        """
        H(Y | F, |Y|=k) in bits, consistent with the SAME combinatorics used elsewhere.

        Here k is fixed (#objects).
        Assumes Y is uniform over all size-k subsets of the full N1*N2 grid.

        Given |Y| = k, Y is uniform over all C(N1*N2, k) bindings.
        For fixed (a,b): choose which a rows and b columns are active,
        and then choose a binding pattern with k edges that covers all of them.
        Hence:
        P(a,b | k) = C(N1, a) * C(N2, b) * C(a, b, k) / C(N1*N2, k)
        The conditional entropy averages the remaining ambiguity over F:
        H(Y | F, |Y|=k) = sum_{a,b} P(a,b | k) * log2 C(a, b, k)

        """
        if self.n_groups != 2:
            raise NotImplementedError("Only supports n_groups == 2 currently.")

        N1, N2 = self.group_sizes[0], self.group_sizes[1]

        n = N1 * N2
        if k < 0 or k > n:
            raise ValueError(f"k must be in [0, {n}].")

        denom = _nCk(n, k)  # total number of Y with |Y|=k
        if denom == 0:
            return 0.0
        if k == 0:
            return 0.0  # only empty set, F fixed, no uncertainty

        H = 0.0
        for a in range(1, N1 + 1):
            if a > k:
                break
            for b in range(1, N2 + 1):
                if b > k:
                    break
                if k > a * b:
                    continue

                C_abk = count_no_empty_groups((a, b), k)
                if C_abk <= 0:
                    continue

                # P(a,b | k) = C(N1,a) C(N2,b) C(a,b,k) / C(N1*N2,k)
                p_ab = (_nCk(N1, a) * _nCk(N2, b) * C_abk) / denom
                H += p_ab * math.log2(C_abk)

        return float(max(H, 0.0))
    

    def _sample_Y(self):
        if self.distribution_mode == "uniform":
            # To sample Y uniformly over all subsets with |Y| ≤ M:
            # there are C(n, m) distinct subsets of size m.
            # So we must sample the size m with probability ∝ C(n, m),
            # otherwise some subsets would be over/under-represented.

            n = self.n_conjunctions
            M = min(self.max_objects, n)

            # 1) sample size m with P(m) ∝ C(n, m)
            weights = torch.tensor([math.comb(n, m) for m in range(M + 1)], dtype=torch.double)
            m = torch.multinomial(weights, 1).item()

            # 2) sample a uniform subset of size m
            Y = torch.zeros(n, dtype=torch.long)
            idx = torch.randperm(n)[:m]
            Y[idx] = 1
            return Y
        raise ValueError(f"Unknown distribution_mode={self.distribution_mode!r}.")

    def generate_dataset(self, n_samples):
        raise NotImplementedError


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [int(p) for p in parts if p != ""]
    return [int(v) for v in list(value)]

DEFAULT_COLORS = {
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (34, 139, 34),
    "yellow": (255, 215, 0),
    "purple": (138, 43, 226),
    "orange": (255, 140, 0),
    "cyan": (0, 206, 209),
    "gray": (128, 128, 128),
}

DEFAULT_SHAPES = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon", "cross", "star"]


def _regular_polygon(cx, cy, radius, n_sides):
    pts = []
    for i in range(n_sides):
        theta = 2.0 * math.pi * i / n_sides - math.pi / 2.0
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts


def _star_polygon(cx, cy, outer_radius, n_points=5, inner_radius=None):
    if inner_radius is None:
        inner_radius = 0.45 * outer_radius
    pts = []
    for i in range(2 * n_points):
        radius = outer_radius if i % 2 == 0 else inner_radius
        theta = math.pi * i / n_points - math.pi / 2.0
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts


def save_binding_dataset_to_disk(
    dataset,
    output_dir: str,
    seed: int,
    save_images: bool = True,
    save_image_metadata: bool = False,
    save_if_missing_only: bool = True,
):
    split = str(getattr(dataset, "split", "data"))
    seed = int(seed) if seed is not None else 0
    dataset_dir = os.path.join(output_dir, f"dataset_{split}_seed{seed}")
    os.makedirs(dataset_dir, exist_ok=True)

    images_dir = None
    if save_images:
        images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    metadata = {
        "Y": dataset.Y,
        "F": dataset.F,
        "H_Y_given_F": dataset.H_Y_given_F,
        "boxes_list": getattr(dataset, "boxes_list", None),
        "n_conjunctions": dataset.n_conjunctions,
        "n_features": dataset.n_features,
        "feature_groups": dataset.feature_groups,
        "group_sizes": dataset.group_sizes,
        "seed": seed,
        "split": split,
        "image_size": getattr(dataset, "image_size", None),
        "num_samples": len(dataset),
        "split_mode": getattr(dataset, "split_mode", None),
        "exclude_all_ones_f": getattr(dataset, "exclude_all_ones_f", None),
        "f_split_hash_mod": getattr(dataset, "f_split_hash_mod", None),
        "f_split_test_buckets": getattr(dataset, "f_split_test_buckets", None),
        "f_split_val_buckets": getattr(dataset, "f_split_val_buckets", None),
        "f_split_salt": getattr(dataset, "f_split_salt", None),
        "f_split_train_count": getattr(dataset, "f_split_train_count", None),
        "f_split_val_count": getattr(dataset, "f_split_val_count", None),
        "f_split_test_count": getattr(dataset, "f_split_test_count", None),
        "split_f_support_count": getattr(dataset, "split_f_support_count", None),
        "total_f_support_count": getattr(dataset, "total_f_support_count", None),
        "f_split_max_attempt_factor": getattr(dataset, "f_split_max_attempt_factor", None),
        "theoretical_H_F": getattr(dataset, "theoretical_H_F", None),
        "theoretical_H_Y_given_F": getattr(dataset, "theoretical_H_Y_given_F", None),
        "theoretical_H_Y": getattr(dataset, "theoretical_H_Y", None),
        "H_F": getattr(dataset, "H_F", None),
        "H_Y": getattr(dataset, "H_Y", None),
        "empirical_H_F": getattr(dataset, "empirical_H_F", None),
        "empirical_H_Y": getattr(dataset, "empirical_H_Y", None),
        "y_support_per_f_count": getattr(dataset, "y_support_per_f_count", None),
        "split_y_support_count": getattr(dataset, "split_y_support_count", None),
        "candidate_colors_per_sample": getattr(dataset, "candidate_colors_per_sample", None),
        "candidate_shapes_per_sample": getattr(dataset, "candidate_shapes_per_sample", None),
        "num_sampling_attempts": getattr(dataset, "num_sampling_attempts", None),
        "num_sampling_rejections": getattr(dataset, "num_sampling_rejections", None),
        "sampling_acceptance_rate": getattr(dataset, "sampling_acceptance_rate", None),
        "num_objects_per_sample": getattr(dataset, "num_objects_per_sample", None),
        "num_objects_histogram": getattr(dataset, "num_objects_histogram", None),
    }
    metadata_path = os.path.join(dataset_dir, "metadata.pt")
    if not (save_if_missing_only and os.path.exists(metadata_path)):
        torch.save(metadata, metadata_path)
        print(f"Dataset metadata saved to {metadata_path}")
    else:
        print(f"Dataset metadata already exists at {metadata_path}; skipping save.")

    if not (save_images or save_image_metadata):
        return

    print(f"Saving dataset artifacts for {len(dataset)} samples...")
    metadata_file = None
    metadata_jsonl_path = os.path.join(dataset_dir, "image_metadata.jsonl")
    extra_image_metadata = getattr(dataset, "image_metadata_extras", None)
    if extra_image_metadata is not None and len(extra_image_metadata) != len(dataset):
        raise ValueError(
            "dataset.image_metadata_extras length must match dataset length: "
            f"{len(extra_image_metadata)} vs {len(dataset)}"
        )
    try:
        if save_image_metadata:
            if save_if_missing_only and os.path.exists(metadata_jsonl_path):
                print(f"Per-image metadata already exists at {metadata_jsonl_path}; skipping save.")
            else:
                metadata_file = open(metadata_jsonl_path, "w")

        for i in range(len(dataset)):
            img, Y, F, H = dataset[i]

            if save_images and images_dir is not None:
                img_path = os.path.join(images_dir, f"sample_{i:06d}.png")
                if not (save_if_missing_only and os.path.exists(img_path)):
                    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
                    img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(img_path)

            if metadata_file is not None:
                boxes = getattr(dataset, "boxes_list", None)
                sample_boxes = boxes[i] if boxes is not None else None
                num_objects = getattr(dataset, "num_objects_per_sample", None)

                entry = {
                    "index": i,
                    "Y": Y.tolist(),
                    "F": F.tolist(),
                    "H_Y_given_F": float(H.item()) if hasattr(H, "item") else float(H),
                    "boxes": sample_boxes,
                }
                if num_objects is not None:
                    v = num_objects[i]
                    entry["num_objects"] = int(v.item()) if hasattr(v, "item") else int(v)

                if extra_image_metadata is not None:
                    extra = extra_image_metadata[i]
                    if extra is not None:
                        if not isinstance(extra, dict):
                            raise TypeError(
                                "dataset.image_metadata_extras must contain dict entries; "
                                f"found {type(extra)} at index {i}"
                            )
                        for k, v in extra.items():
                            if k in entry:
                                continue
                            entry[k] = v

                metadata_file.write(
                    json.dumps(entry)
                    + "\n"
                )
    finally:
        if metadata_file is not None:
            metadata_file.close()

    if save_images and images_dir is not None:
        print(f"Dataset images saved to {images_dir}")
    if save_image_metadata and os.path.exists(metadata_jsonl_path):
        print(f"Per-image metadata saved to {metadata_jsonl_path}")



class ColorShapeDataset(BindingDataset):
    def __init__(self, cfg, seed=0, num_samples=None, transform=None, output_dir=None, split="train"):
        # ----------------------------
        # feature space
        # ----------------------------
        n_colors = int(cfg.dataset.get("n_colors", 2))
        n_shapes = int(cfg.dataset.get("n_shapes", 2))

        colors = list(DEFAULT_COLORS.keys())[:n_colors]
        shapes = list(DEFAULT_SHAPES)[:n_shapes]

        super().__init__(
            feature_groups=[colors, shapes],
            distribution_mode=cfg.dataset.get("distribution_mode", "uniform"),
            max_objects=cfg.dataset.get("max_objects", None),
            transform=transform,
        )

        # ----------------------------
        # randomness
        # ----------------------------
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.seed = seed
        self.split = split

        # ----------------------------
        # optional split policy for F-disjoint test
        # ----------------------------
        self.split_mode = str(cfg.dataset.get("split_mode", "independent")).strip().lower()
        self.exclude_all_ones_f = bool(cfg.dataset.get("exclude_all_ones_f", False))
        self.f_split_hash_mod = int(cfg.dataset.get("f_split_hash_mod", 10))
        self.f_split_test_buckets = sorted(set(_parse_int_list(cfg.dataset.get("f_split_test_buckets", [0]))))
        self.f_split_val_buckets = sorted(set(_parse_int_list(cfg.dataset.get("f_split_val_buckets", []))))
        self.f_split_test_bucket_set = set(self.f_split_test_buckets)
        self.f_split_val_bucket_set = set(self.f_split_val_buckets)
        self.f_split_salt = str(cfg.dataset.get("f_split_salt", "colorshape_fsplit_v1"))
        self.f_split_max_attempt_factor = int(cfg.dataset.get("f_split_max_attempt_factor", 200))
        self.f_split_train_count = cfg.dataset.get("f_split_train_count", None)
        self.f_split_val_count = cfg.dataset.get("f_split_val_count", None)
        self.f_split_test_count = cfg.dataset.get("f_split_test_count", None)
        self.f_split_train_count = None if self.f_split_train_count is None else int(self.f_split_train_count)
        self.f_split_val_count = None if self.f_split_val_count is None else int(self.f_split_val_count)
        self.f_split_test_count = None if self.f_split_test_count is None else int(self.f_split_test_count)

        valid_split_modes = {"independent", "f_disjoint_hash", "f_ranked_three_way"}
        if self.split_mode not in valid_split_modes:
            raise ValueError(
                f"Unknown dataset.split_mode={self.split_mode!r}. "
                f"Expected one of: {sorted(valid_split_modes)}"
            )
        if self.split_mode == "f_disjoint_hash":
            if self.f_split_hash_mod <= 1:
                raise ValueError("dataset.f_split_hash_mod must be > 1 in f_disjoint_hash mode.")
            if len(self.f_split_test_buckets) == 0:
                raise ValueError("dataset.f_split_test_buckets must be non-empty in f_disjoint_hash mode.")
            bad = [b for b in self.f_split_test_buckets if b < 0 or b >= self.f_split_hash_mod]
            if bad:
                raise ValueError(
                    f"dataset.f_split_test_buckets contains out-of-range buckets {bad}; "
                    f"valid range is [0, {self.f_split_hash_mod - 1}]."
                )
            bad_val = [b for b in self.f_split_val_buckets if b < 0 or b >= self.f_split_hash_mod]
            if bad_val:
                raise ValueError(
                    f"dataset.f_split_val_buckets contains out-of-range buckets {bad_val}; "
                    f"valid range is [0, {self.f_split_hash_mod - 1}]."
                )
        self._split_name_lc = str(self.split).strip().lower()
        self._all_ones_f_tuple = tuple(1 for _ in range(self.n_features))

        # ----------------------------
        # rendering config
        # ----------------------------
        self.image_size = int(cfg.dataset.get("image_size", 128))
        self.obj_size = int(cfg.dataset.get("obj_size", 28))
        self.margin = int(cfg.dataset.get("margin", 4))
        self.max_place_tries = int(cfg.dataset.get("max_place_tries", 2000))
        self.non_overlapping = bool(cfg.dataset.get("non_overlapping", True))
        self.background = tuple(cfg.dataset.get("background", (255, 255, 255)))

        # color palette (restricted to chosen colors)
        self.color_rgb = {c: DEFAULT_COLORS[c] for c in colors}

        self.num_samples = num_samples

        self.generate_dataset(num_samples if num_samples is not None else 1000)
        should_save_images = bool(cfg.dataset.get("save_dataset", False))
        should_save_metadata = bool(cfg.dataset.get("save_image_metadata", False))
        save_if_missing_only = bool(cfg.dataset.get("save_if_missing_only", True))
        if output_dir is not None and (should_save_images or should_save_metadata):
            save_binding_dataset_to_disk(
                self,
                output_dir=output_dir,
                seed=seed,
                save_images=should_save_images,
                save_image_metadata=should_save_metadata,
                save_if_missing_only=save_if_missing_only,
            )

    def _boxes_overlap(self, a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    def _place_objects(self, n_obj):
        H = W = self.image_size
        s = self.obj_size
        m = self.margin
        boxes = []
        for _ in range(n_obj):
            ok = False
            for _try in range(self.max_place_tries):
                x0 = random.randint(m, W - m - s)
                y0 = random.randint(m, H - m - s)
                box = (x0, y0, x0 + s, y0 + s)
                if not self.non_overlapping:
                    boxes.append(box)
                    ok = True
                    break
                if all(not self._boxes_overlap(box, b) for b in boxes):
                    boxes.append(box)
                    ok = True
                    break
            if not ok:
                raise RuntimeError("Failed to place non-overlapping objects; reduce obj_size or increase image_size.")
        return boxes

    def _render(self, Y, boxes):
        img = Image.new("RGB", (self.image_size, self.image_size), self.background)
        draw = ImageDraw.Draw(img)

        active = torch.nonzero(Y > 0, as_tuple=False).view(-1).tolist()
        if len(active) != len(boxes):
            raise ValueError("boxes must match #active objects")

        for i, conj_id in enumerate(active):
            # conj_id -> (color_idx, shape_idx)
            color_i, shape_i = self.conjunction_group_indices[conj_id]
            color = self.feature_groups[0][color_i]
            shape = self.feature_groups[1][shape_i]

            fill = self.color_rgb[color]
            x0, y0, x1, y1 = boxes[i]
            w, h = (x1 - x0), (y1 - y0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            radius = 0.5 * min(w, h)

            if shape == "circle":
                draw.ellipse([x0, y0, x1, y1], fill=fill)
            elif shape == "square":
                draw.rectangle([x0, y0, x1, y1], fill=fill)
            elif shape == "triangle":
                pts = _regular_polygon(cx, cy, radius, 3)
                draw.polygon(pts, fill=fill)
            elif shape == "diamond":
                pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
                draw.polygon(pts, fill=fill)
            elif shape == "pentagon":
                pts = _regular_polygon(cx, cy, radius, 5)
                draw.polygon(pts, fill=fill)
            elif shape == "hexagon":
                pts = _regular_polygon(cx, cy, radius, 6)
                draw.polygon(pts, fill=fill)
            elif shape == "cross":
                t = max(2, int(min(w, h) * 0.25))
                # vertical bar
                draw.rectangle([int(cx - t / 2), y0, int(cx + t / 2), y1], fill=fill)
                # horizontal bar
                draw.rectangle([x0, int(cy - t / 2), x1, int(cy + t / 2)], fill=fill)
            elif shape == "star":
                pts = _star_polygon(cx, cy, radius, n_points=5)
                draw.polygon(pts, fill=fill)
            else:
                raise ValueError("Unknown shape: %s" % shape)

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _all_feature_vectors(self):
        vectors = []
        for bits in itertools.product([0, 1], repeat=self.n_features):
            f = torch.tensor(bits, dtype=torch.long)
            counts = []
            for g in range(self.n_groups):
                off = self.group_offsets[g]
                sz = self.group_sizes[g]
                counts.append(int(f[off : off + sz].sum().item()))
            if all(c == 0 for c in counts) or all(c > 0 for c in counts):
                vectors.append(f)
        return vectors

    def generate_dataset(self, n_samples):
        boxes_list, Ys, Fs, Hs = [], [], [], []
        target = int(n_samples)
        attempts = 0
        max_attempts = max(target, int(target * max(1, self.f_split_max_attempt_factor)))
        while len(Ys) < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Could not generate enough samples under split constraints. "
                    f"split={self.split}, split_mode={self.split_mode}, target={target}, "
                    f"accepted={len(Ys)}, attempts={attempts}, max_attempts={max_attempts}. "
                    "Increase dataset.f_split_max_attempt_factor or relax split buckets."
                )
            y = self._sample_Y()
            f = self.Y_to_F(y)
            if not self._accept_f_for_split(f):
                continue

            n_obj = int(y.sum().item())
            boxes = self._place_objects(n_obj)
            h = self.get_H_Y_F(f)

            boxes_list.append(boxes)
            Ys.append(y)
            Fs.append(f)
            Hs.append(h)

        self.boxes_list = boxes_list
        self.Y = torch.stack(Ys, 0).contiguous()
        self.F = torch.stack(Fs, 0).contiguous()
        self.H_Y_given_F = torch.stack(Hs, 0).contiguous()
        self.mean_H_Y_given_F = self.H_Y_given_F.mean().item()
        self.num_sampling_attempts = int(attempts)
        self.num_sampling_rejections = int(attempts - target)
        self.sampling_acceptance_rate = float(target / attempts) if attempts > 0 else 0.0

    def __getitem__(self, idx):
        img = self._render(self.Y[idx], self.boxes_list[idx])
        if getattr(self, "transform", None) is not None:
            img = self.transform(img)
        return img, self.Y[idx], self.F[idx], self.H_Y_given_F[idx]

    def _f_bucket(self, f: torch.Tensor) -> int:
        bits = "".join(str(int(v)) for v in f.view(-1).tolist())
        key = f"{self.f_split_salt}|{bits}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % int(self.f_split_hash_mod)

    def _f_tuple(self, f: torch.Tensor):
        return tuple(int(v) for v in f.view(-1).tolist())

    def _ranked_f_split_candidates(self):
        return self._all_feature_vectors()

    def _ranked_f_split_tables(self):
        if hasattr(self, "_ranked_f_split_table_cache"):
            return self._ranked_f_split_table_cache

        candidates = []
        seen = set()
        for f in self._ranked_f_split_candidates():
            f_tuple = self._f_tuple(f)
            if self.exclude_all_ones_f and f_tuple == self._all_ones_f_tuple:
                continue
            if f_tuple in seen:
                continue
            seen.add(f_tuple)
            bits = "".join(str(v) for v in f_tuple)
            key = f"{self.f_split_salt}|{bits}"
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            candidates.append((digest, bits, f_tuple))
        candidates.sort(key=lambda item: (item[0], item[1]))

        total = len(candidates)
        if total == 0:
            raise RuntimeError(
                f"No F candidates are available for split_mode={self.split_mode!r}."
            )

        val_count = self.f_split_val_count
        test_count = self.f_split_test_count
        train_count = self.f_split_train_count
        if val_count is None:
            val_count = total // 6
        if test_count is None:
            test_count = total // 6
        if train_count is None:
            train_count = total - int(val_count) - int(test_count)

        train_count = int(train_count)
        val_count = int(val_count)
        test_count = int(test_count)
        if min(train_count, val_count, test_count) < 0:
            raise ValueError("f_split_train_count/val_count/test_count must be non-negative.")
        if train_count + val_count + test_count != total:
            raise ValueError(
                "Ranked F split counts must sum to the available F support. "
                f"Got train/val/test={train_count}/{val_count}/{test_count}, total={total}."
            )

        split_by_tuple = {}
        f_by_split = {"train": [], "val": [], "test": []}
        for _digest, _bits, f_tuple in candidates[:train_count]:
            split_by_tuple[f_tuple] = "train"
            f_by_split["train"].append(f_tuple)
        val_start = train_count
        val_end = train_count + val_count
        for _digest, _bits, f_tuple in candidates[val_start:val_end]:
            split_by_tuple[f_tuple] = "val"
            f_by_split["val"].append(f_tuple)
        for _digest, _bits, f_tuple in candidates[val_end:]:
            split_by_tuple[f_tuple] = "test"
            f_by_split["test"].append(f_tuple)

        self._ranked_f_split_table_cache = {
            "split_by_tuple": split_by_tuple,
            "f_by_split": f_by_split,
            "total": total,
        }
        return self._ranked_f_split_table_cache

    def _ranked_f_split_for_f(self, f: torch.Tensor):
        tables = self._ranked_f_split_tables()
        return tables["split_by_tuple"].get(self._f_tuple(f), None)

    def _accept_f_for_split(self, f: torch.Tensor) -> bool:
        if self.exclude_all_ones_f:
            f_tuple = self._f_tuple(f)
            if f_tuple == self._all_ones_f_tuple:
                return False
        if self.split_mode == "f_ranked_three_way":
            f_split = self._ranked_f_split_for_f(f)
            if self._split_name_lc == "test":
                return f_split == "test"
            if self._split_name_lc == "val":
                return f_split == "val"
            return f_split == "train"
        if self.split_mode != "f_disjoint_hash":
            return True
        bucket = self._f_bucket(f)
        in_test_set = bucket in self.f_split_test_bucket_set
        if self._split_name_lc == "test":
            return in_test_set
        if self._split_name_lc == "val" and len(self.f_split_val_bucket_set) > 0:
            return bucket in self.f_split_val_bucket_set
        if len(self.f_split_val_bucket_set) > 0:
            return (not in_test_set) and (bucket not in self.f_split_val_bucket_set)
        return not in_test_set


# quick test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(repo_root, "data", "generated", "colorshape_main_test")
    os.makedirs(output_dir, exist_ok=True)

    raw_cfg = {
        "dataset": {
            "n_colors": 7,
            "n_shapes": 7,
            "image_size": 224,
            "obj_size": 16,
            "distribution_mode": "uniform",
            "non_overlapping": True,
            "seed": 0,
            "max_objects": 24
        }
    }

    cfg = OmegaConf.create(raw_cfg)
    ds = ColorShapeDataset(cfg, num_samples=5)
    for i in range(5):
        img, Y, F, H = ds[i]

        print("Y:", Y.tolist())
        print("F:", F.tolist())
        print("H(Y|F):", float(H.item()))

        img_np = img.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        
        plt.title(f"H(Y|F) = {H.item():.2f} bits")
        plt.savefig(os.path.join(output_dir, f"test_colorshape_{i}.png"), bbox_inches="tight")
        plt.close()

    H_k_values = []
    for k in range(1, 50):
        H_k = ds.get_H_Y_F_given_num_objects(k)
        print(f"H(Y|F, |Y|={k}) = {H_k:.4f} bits")
        H_k_values.append(H_k)
    plt.figure()
    plt.plot(range(1, 50), H_k_values, marker="o")
    plt.xlabel("|Y| = number of objects")
    plt.ylabel("H(Y|F, |Y|) in bits")
    plt.title("Conditional Entropy vs Number of Objects")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "test_colorshape_H_num_objects.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved ColorShape smoke-test outputs to {output_dir}")
    

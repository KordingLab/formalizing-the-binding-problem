import os
import json
import math
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from .ColorShape import BindingDataset

import numpy as np
from collections import defaultdict

from tqdm import tqdm

from functools import lru_cache

def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _basename_and_root_from_url(url: str):
    # e.g. ".../VG_100K_2/1.jpg" -> ("VG_100K_2", "1.jpg")
    parts = url.split("/")
    return parts[-2], parts[-1]


def _pil_to_chw_float(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)



class VGColorDataset(BindingDataset):
    """
    Visual Genome color binding dataset built from mined json/jsonl.
    """

    def __init__(self, cfg, seed=0, num_samples=None, transform=None, output_dir=None, split="train"):
        del output_dir
        self.cfg = cfg
        self.seed = seed if seed is not None else 0
        self.split = str(split)
        self.transform = transform
        self.data_dir = str(cfg.dataset.data_dir)
        self.image_size = int(cfg.dataset.get("image_size", 224))
        metadata_split = str(cfg.dataset.get("metadata_split", cfg.dataset.get("split", "coco")))
        meta_dir = os.path.join(self.data_dir, "deprecated", f"meta_filtered_color_{metadata_split}")

        # ---- load files ----
        meta_path = os.path.join(meta_dir, "attributes_filtered_color.json")
        cat_attr_path = os.path.join(meta_dir, "vg_stats_filtered_color.json")
        split_path = os.path.join(meta_dir, "split_dataset.json")  # optional

        cat_attr = _read_json(cat_attr_path)
        all_meta = _read_json(meta_path)

        # ---- categories/colors from stats ----
        # expects vg_stats_filtered_color.json produced by earlier script
        categories = cat_attr.get("selected_object_classes", [])
        colors = cat_attr.get("selected_color_attributes", [])

        if "no_color" not in colors:
            colors = list(colors) + ["no_color"]

        # keep stable order: most common first if present
        obj_occ = cat_attr.get("selected_object_occurrences", {})
        attr_occ = cat_attr.get("selected_attribute_occurrences", {})

        if isinstance(obj_occ, dict) and len(obj_occ) > 0:
            categories = sorted(categories, key=lambda c: (-obj_occ.get(c, 0), c))
        if isinstance(attr_occ, dict) and len(attr_occ) > 0:
            colors = sorted(colors, key=lambda a: (-attr_occ.get(a, 0), a))

        self.categories = categories
        self.colors = colors
        self.cat2i = {c: i for i, c in enumerate(categories)}
        self.col2i = {c: i for i, c in enumerate(colors)}

        super().__init__(
            feature_groups=[categories, colors],
            distribution_mode="natural",
            max_objects=None,
            transform=transform,
        )

        # ---- load ids + metadata (apply split if split file exists) ----
        split_ids = None
        if os.path.exists(split_path):
            split_dict = _read_json(split_path)
            split_ids = set(split_dict.get(split, []))

        if split_ids is not None:
            self.all_metadata = [m for m in all_meta if m.get("image_id") in split_ids]
        else:
            self.all_metadata = all_meta
        if split_ids is None and num_samples is not None:
            train_total = int(cfg.dataset.get("num_train_samples", 0))
            train_count = int(train_total * 0.8)
            val_count = int(train_total * 0.2)
            start = train_count if self.split == "val" else train_count + val_count if self.split == "test" else 0
            self.all_metadata = self.all_metadata[start : start + int(num_samples)]
        else:
            if seed is not None:
                rng = np.random.default_rng(int(seed))
                order = rng.permutation(len(self.all_metadata))
                self.all_metadata = [self.all_metadata[int(i)] for i in order]
            if num_samples is not None:
                self.all_metadata = self.all_metadata[: int(num_samples)]

        self.image_ids = [m.get("image_id") for m in self.all_metadata]

        # ---- conjunction statistics: [N1, N2] from vg_stats_filtered_color ----
        N1, N2 = len(categories), len(colors)
        self.conjunction_statistics = torch.zeros((N1, N2), dtype=torch.float64)

        obj2attr = cat_attr.get("object_to_color_attribute_occurrences", {})  # {obj: {attr: count}}
        total = 0.0
        if isinstance(obj2attr, dict):
            for cat, attr_counts in obj2attr.items():
                if cat not in self.cat2i or not isinstance(attr_counts, dict):
                    continue
                i = self.cat2i[cat]
                for a, cnt in attr_counts.items():
                    if a not in self.col2i:
                        continue
                    j = self.col2i[a]
                    c = float(cnt)
                    self.conjunction_statistics[i, j] += c
                    total += c

        self._total_conj = float(total) if total > 0 else 1.0
        self._eps = 1e-12

        # ---- build per-image (F,Y,H) ----
        self.F, self.Y, self.H_Y_given_F = [], [], []
        for m in tqdm(self.all_metadata):
            Y, F, H = metadata_to_features(
                m,
                categories=self.categories,
                colors=self.colors,
                cat2i=self.cat2i,
                col2i=self.col2i,
                conj_counts=self.conjunction_statistics,
                total_conj=self._total_conj,
                eps=self._eps,
            )
            self.Y.append(Y)
            self.F.append(F)
            self.H_Y_given_F.append(H)

        if len(self.Y) == 0:
            raise RuntimeError(f"No VGColor samples found for split={self.split!r} under {meta_dir}.")
        self.Y = torch.stack(self.Y, 0).contiguous()
        self.F = torch.stack(self.F, 0).contiguous()
        self.H_Y_given_F = torch.tensor(self.H_Y_given_F, dtype=torch.float32).contiguous()
        self.mean_H_Y_given_F = float(self.H_Y_given_F.mean().item())

        # ---- image root ----
        self.img_root = getattr(cfg.dataset, "img_root", self.data_dir)

    def __len__(self):
        return len(self.all_metadata)

    def __getitem__(self, idx):
        m = self.all_metadata[idx]        
        
        root, fname = _basename_and_root_from_url(m["image_url"])
        img_path = os.path.join(self.img_root, root, fname)
        image = Image.open(img_path).convert("RGB")
        img = _pil_to_chw_float(image)
        if getattr(self, "transform", None) is not None:
            img = self.transform(img)
        return img, self.Y[idx], self.F[idx], self.H_Y_given_F[idx]



def metadata_to_features(
    metadata: dict,
    *,
    categories: List[str],
    colors: List[str],
    cat2i: Dict[str, int],
    col2i: Dict[str, int],
    conj_counts: torch.Tensor,   # [N1, N2]
    total_conj: float,
    eps: float = 1e-12,
    verbose: bool = False,
    max_enum: int = 50_000,
    feasibility_threshold: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    VGColor: instance-level categories + colors, binary Y and F, and stable log-space H(Y|F).

    IMPORTANT INVARIANT ENFORCED:
      F is derived from Y (marginals), so every active category/color in F participates
      in at least one active (category,color) pair in Y.
    """
    def vprint(*args):
        if verbose:
            print(*args)
    
    N1, N2 = len(categories), len(colors)
    NO_COLOR = "no_attribute" if "no_attribute" in col2i else "no_color"

    # ---- read objects ----
    objs = metadata.get("attributes", []) or []

    # ---- collect instances (one category + one chosen color per instance) ----
    Cinst: List[str] = []
    Ainst: List[str] = []

    for o in objs:
        c = o.get("object_class")
        if c is None or c not in cat2i:
            continue

        attrs = o.get("attributes_color", []) or []
        chosen = None
        for a in attrs:
            if a in col2i and a != NO_COLOR:
                chosen = a
                break
        if chosen is None or chosen not in col2i:
            chosen = NO_COLOR

        Cinst.append(c)
        Ainst.append(chosen)

    m = len(Cinst)
    if m == 0:
        return torch.zeros(N1 * N2), torch.zeros(N1 + N2), 0.0

    # ---- Y: binary presence of (category,color) pairs in observed metadata ----
    Ymat = torch.zeros((N1, N2), dtype=torch.float32)
    for c, a in zip(Cinst, Ainst):
        a = a if a in col2i else NO_COLOR
        Ymat[cat2i[c], col2i[a]] = 1.0
    Y = Ymat.reshape(-1)

    # ---- F: DERIVED FROM Y (marginals) ----
    # categories present iff any color binds to it
    # colors present iff any category binds to it
    F = torch.zeros(N1 + N2, dtype=torch.float32)
    F[:N1] = (Ymat.sum(dim=1) > 0).to(torch.float32)   # [N1]
    F[N1:] = (Ymat.sum(dim=0) > 0).to(torch.float32)   # [N2]

    # ---- p(c,a) from dataset stats ----
    denom_total = float(total_conj) if float(total_conj) > 0 else 1.0

    def p_ca(cat: str, attr: str) -> float:
        return float(conj_counts[cat2i[cat], col2i[attr]].item()) / denom_total

    # ---- color multiset (supply) from observed instance colors ----
    # Note: this supply is from the kept (cat2i-filtered) instances only.
    color_counts = defaultdict(int)
    for a in Ainst:
        a = a if a in col2i else NO_COLOR
        color_counts[a] += 1

    # Make ordering deterministic (helps reproducibility / debugging)
    color_keys = sorted(list(color_counts.keys()), key=lambda x: col2i.get(x, 10**9))
    key2idx = {a: i for i, a in enumerate(color_keys)}
    init_supply = tuple(color_counts[a] for a in color_keys)

    # naive multiset permutation count (ignores feasibility)
    denom = 1
    for a in color_keys:
        denom *= math.factorial(color_counts[a])
    num_assignments_naive = math.factorial(m) // denom

    vprint("Instances:")
    for i, (c, a) in enumerate(zip(Cinst, Ainst)):
        vprint(f"  #{i}: ({c}, {a})")
    vprint("Colors present (A):", color_keys)
    vprint("Color multiset:", dict(color_counts))
    vprint("Naive multiset perms:", num_assignments_naive)

    # allowed colors per instance (currently: any present color)
    allowed_idx: List[Tuple[int, ...]] = [tuple(range(len(color_keys))) for _ in range(m)]

    # ---- log-space DP for entropy over assignments ----
    LOG2 = math.log(2.0)

    def logsumexp(xs: List[float]) -> float:
        mx = max(xs)
        if mx == -float("inf"):
            return -float("inf")
        return mx + math.log(sum(math.exp(x - mx) for x in xs))

    def logp_ca(cat: str, attr: str) -> float:
        p = p_ca(cat, attr)
        return math.log(p + eps)

    @lru_cache(maxsize=None)
    def dp_logZ_Elogw(i: int, supply: tuple[int, ...]) -> Tuple[float, float]:
        """
        Returns (logZ, Elogw) in natural log:
          logZ  = log(sum_w) over valid assignments for instances i..m-1
          Elogw = E_P[ log(w_suffix) ] where P is normalized over suffix assignments
        """
        if i == m:
            return (0.0, 0.0) if sum(supply) == 0 else (-float("inf"), 0.0)

        c = Cinst[i]
        terms = []   # t = logp + child_logZ
        kids = []    # (t, logp, child_Elogw)

        for j in allowed_idx[i]:
            if supply[j] == 0:
                continue
            a = color_keys[j]
            lp = logp_ca(c, a)

            s = list(supply)
            s[j] -= 1
            s2 = tuple(s)

            child_logZ, child_E = dp_logZ_Elogw(i + 1, s2)
            if child_logZ == -float("inf"):
                continue

            t = lp + child_logZ
            terms.append(t)
            kids.append((t, lp, child_E))

        if not terms:
            return (-float("inf"), 0.0)

        logZ = logsumexp(terms)

        # E[log w] = sum_branch P(branch) * (logp + child_Elogw)
        Elogw = 0.0
        for t, lp, child_E in kids:
            p_branch = math.exp(t - logZ)
            Elogw += p_branch * (lp + child_E)

        return (logZ, Elogw)

    logZ, Elogw = dp_logZ_Elogw(0, init_supply)
    if logZ == -float("inf"):
        vprint("logZ=-inf (no feasible assignments); returning H=0.0")
        return Y, F, 0.0

    # H = (-E[log w] + log Z) / ln 2
    H = (-Elogw + logZ) / LOG2
    vprint("logZ:", logZ, "Elogw:", Elogw, "H:", H)
    
    return Y, F, float(H)




def collate_fn_vg(batch):
    """
    Batch elements:
      image: PIL.Image
      F: Tensor [N1+N2]
      Y: Tensor [N1*N2]
      H: float
    """
    images, Ys, Fs, Hs = zip(*batch)

    # images stay as a list (common for vision pipelines / transforms later)
    images = list(images)

    Ys = torch.stack(Ys, dim=0)   # [B, N1*N2]
    Fs = torch.stack(Fs, dim=0)   # [B, N1+N2]
    Hs = torch.tensor(Hs, dtype=torch.float32)  # [B]

    return images, Ys, Fs, Hs

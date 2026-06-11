import itertools
import math
import random
from collections import Counter

import torch

from .ColorShape import ColorShapeDataset, count_no_empty_groups


class BalancedColorShapeDataset(ColorShapeDataset):
    """
    ColorShape variant with fixed feature and object support.

    Each sample chooses a fixed-size feature code F: six colors and six shapes from
    an 8x8 feature space. It then samples 18 distinct color-shape objects from the
    selected 6x6 grid, rejecting draws that do not cover every selected feature.
    """

    def __init__(self, cfg, seed=0, num_samples=None, transform=None, output_dir=None, split="train"):
        ds_cfg = cfg.dataset
        self.candidate_colors_per_sample = int(ds_cfg.get("candidate_colors_per_sample", 6))
        self.candidate_shapes_per_sample = int(ds_cfg.get("candidate_shapes_per_sample", 6))
        self.num_objects = int(ds_cfg.get("num_objects", 18))
        self.object_sample_max_attempts = int(ds_cfg.get("object_sample_max_attempts", 1000))
        super().__init__(
            cfg=cfg,
            seed=seed,
            num_samples=num_samples,
            transform=transform,
            output_dir=output_dir,
            split=split,
        )

    def _ranked_f_split_candidates(self):
        out = []
        n_colors = len(self.feature_groups[0])
        n_shapes = len(self.feature_groups[1])
        color_offset = self.group_offsets[0]
        shape_offset = self.group_offsets[1]

        for color_idxs in itertools.combinations(range(n_colors), self.candidate_colors_per_sample):
            for shape_idxs in itertools.combinations(range(n_shapes), self.candidate_shapes_per_sample):
                f = torch.zeros(self.n_features, dtype=torch.long)
                for idx in color_idxs:
                    f[color_offset + int(idx)] = 1
                for idx in shape_idxs:
                    f[shape_offset + int(idx)] = 1
                out.append(f)
        return out

    def _split_key(self):
        if self._split_name_lc == "test":
            return "test"
        if self._split_name_lc == "val":
            return "val"
        return "train"

    def _allowed_f_candidates_for_split(self):
        if hasattr(self, "_balanced_allowed_f_candidates"):
            return self._balanced_allowed_f_candidates

        if self.split_mode == "f_ranked_three_way":
            tables = self._ranked_f_split_tables()
            f_tuples = tables["f_by_split"][self._split_key()]
            self.total_f_support_count = int(tables["total"])
        else:
            f_tuples = [
                self._f_tuple(f)
                for f in self._ranked_f_split_candidates()
                if self._accept_f_for_split(f)
            ]
            self.total_f_support_count = len(self._ranked_f_split_candidates())

        candidates = [torch.tensor(f_tuple, dtype=torch.long) for f_tuple in f_tuples]
        if len(candidates) == 0:
            raise RuntimeError(
                "No BalancedColorShape F candidates are available for this split. "
                f"split={self.split}, split_mode={self.split_mode}."
            )

        self.split_f_support_count = int(len(candidates))
        self._balanced_allowed_f_candidates = candidates
        return self._balanced_allowed_f_candidates

    def _conj_id(self, color_idx: int, shape_idx: int) -> int:
        n_shapes = len(self.feature_groups[1])
        return int(color_idx) * n_shapes + int(shape_idx)

    def _active_feature_indices(self, f):
        color_offset = self.group_offsets[0]
        shape_offset = self.group_offsets[1]
        n_colors = len(self.feature_groups[0])
        n_shapes = len(self.feature_groups[1])
        colors = [
            i
            for i in range(n_colors)
            if int(f[color_offset + i].item()) == 1
        ]
        shapes = [
            i
            for i in range(n_shapes)
            if int(f[shape_offset + i].item()) == 1
        ]
        return colors, shapes

    def _sample_y_covering_f(self, f):
        colors, shapes = self._active_feature_indices(f)
        if len(colors) != self.candidate_colors_per_sample:
            raise ValueError(
                f"Expected {self.candidate_colors_per_sample} active colors, got {len(colors)}."
            )
        if len(shapes) != self.candidate_shapes_per_sample:
            raise ValueError(
                f"Expected {self.candidate_shapes_per_sample} active shapes, got {len(shapes)}."
            )

        all_pairs = [(c, s, self._conj_id(c, s)) for c in colors for s in shapes]
        if self.num_objects > len(all_pairs):
            raise ValueError(
                f"num_objects={self.num_objects} exceeds selected grid size {len(all_pairs)}."
            )

        for attempt in range(1, self.object_sample_max_attempts + 1):
            selected = random.sample(all_pairs, self.num_objects)
            covered_colors = {int(c) for c, _s, _cid in selected}
            covered_shapes = {int(s) for _c, s, _cid in selected}
            if len(covered_colors) != len(colors) or len(covered_shapes) != len(shapes):
                continue

            y = torch.zeros(self.n_conjunctions, dtype=torch.long)
            object_conj_ids = sorted(int(cid) for _c, _s, cid in selected)
            y[torch.tensor(object_conj_ids, dtype=torch.long)] = 1
            return y, object_conj_ids, int(attempt)

        raise RuntimeError(
            "Failed to sample a coverage-valid BalancedColorShape object code. "
            f"Increase object_sample_max_attempts (currently {self.object_sample_max_attempts})."
        )

    def _validate_balanced_config(self):
        if self.n_groups != 2:
            raise ValueError("BalancedColorShapeDataset requires exactly two feature groups.")
        if self.candidate_colors_per_sample < 1 or self.candidate_shapes_per_sample < 1:
            raise ValueError("candidate_colors_per_sample and candidate_shapes_per_sample must be positive.")
        if self.candidate_colors_per_sample > len(self.feature_groups[0]):
            raise ValueError("candidate_colors_per_sample exceeds the configured color count.")
        if self.candidate_shapes_per_sample > len(self.feature_groups[1]):
            raise ValueError("candidate_shapes_per_sample exceeds the configured shape count.")
        if self.num_objects < max(self.candidate_colors_per_sample, self.candidate_shapes_per_sample):
            raise ValueError("num_objects is too small to cover every selected color and shape.")
        grid_size = self.candidate_colors_per_sample * self.candidate_shapes_per_sample
        if self.num_objects > grid_size:
            raise ValueError("num_objects exceeds the selected candidate grid size.")
        if self.max_objects is not None and int(self.max_objects) < self.num_objects:
            raise ValueError("dataset.max_objects must be >= dataset.num_objects.")
        if self.object_sample_max_attempts < 1:
            raise ValueError("object_sample_max_attempts must be >= 1.")

    def generate_dataset(self, n_samples):
        self._validate_balanced_config()
        target = int(n_samples)
        if target < 1:
            raise ValueError("n_samples must be >= 1.")

        y_support_count = count_no_empty_groups(
            (self.candidate_colors_per_sample, self.candidate_shapes_per_sample),
            self.num_objects,
        )
        if y_support_count <= 0:
            raise ValueError(
                "No coverage-valid object codes exist for the configured candidate feature counts "
                f"and num_objects={self.num_objects}."
            )

        allowed_f = self._allowed_f_candidates_for_split()
        h_y_given_f = float(math.log2(y_support_count))
        self.y_support_per_f_count = int(y_support_count)
        self.theoretical_H_Y_given_F = h_y_given_f
        self.theoretical_H_F = float(math.log2(len(allowed_f)))
        self.theoretical_H_Y = float(self.theoretical_H_F + self.theoretical_H_Y_given_F)
        self.H_F = self.theoretical_H_F
        self.H_Y = self.theoretical_H_Y
        self.split_y_support_count = int(len(allowed_f) * y_support_count)

        boxes_list, Ys, Fs, Hs = [], [], [], []
        image_metadata_extras = []
        num_objects_per_sample = []
        sampled_f_counts = Counter()
        sampled_y_counts = Counter()
        total_object_sampling_attempts = 0

        for _sample_idx in range(target):
            f = allowed_f[torch.randint(len(allowed_f), (1,)).item()].clone()
            y, object_conj_ids, object_sampling_attempts = self._sample_y_covering_f(f)
            total_object_sampling_attempts += int(object_sampling_attempts)
            f_from_y = self.Y_to_F(y)
            if not torch.equal(f, f_from_y):
                raise AssertionError("BalancedColorShape sampled Y does not cover exactly the chosen F.")

            boxes = self._place_objects(self.num_objects)
            h = torch.tensor(h_y_given_f, dtype=torch.float32)
            color_idxs, shape_idxs = self._active_feature_indices(f)

            boxes_list.append(boxes)
            Ys.append(y)
            Fs.append(f)
            Hs.append(h)
            num_objects_per_sample.append(int(self.num_objects))

            f_key = self._f_tuple(f)
            y_key = tuple(int(v) for v in y.view(-1).tolist())
            sampled_f_counts[f_key] += 1
            sampled_y_counts[y_key] += 1

            image_metadata_extras.append({
                "selected_color_indices": [int(i) for i in color_idxs],
                "selected_shape_indices": [int(i) for i in shape_idxs],
                "selected_color_names": [self.feature_groups[0][int(i)] for i in color_idxs],
                "selected_shape_names": [self.feature_groups[1][int(i)] for i in shape_idxs],
                "object_conjunction_ids": [int(i) for i in object_conj_ids],
            })

        self.boxes_list = boxes_list
        self.Y = torch.stack(Ys, 0).contiguous()
        self.F = torch.stack(Fs, 0).contiguous()
        self.H_Y_given_F = torch.stack(Hs, 0).contiguous()
        self.mean_H_Y_given_F = self.H_Y_given_F.mean().item()
        self.image_metadata_extras = image_metadata_extras
        self.num_objects_per_sample = torch.tensor(num_objects_per_sample, dtype=torch.long)
        self.num_objects_histogram = {int(k): int(v) for k, v in sorted(Counter(num_objects_per_sample).items())}

        self.num_sampling_attempts = int(total_object_sampling_attempts)
        self.num_sampling_rejections = int(total_object_sampling_attempts - target)
        self.sampling_acceptance_rate = (
            float(target / total_object_sampling_attempts)
            if total_object_sampling_attempts > 0
            else 0.0
        )
        self.empirical_H_F = self._empirical_entropy_from_counter(sampled_f_counts)
        self.empirical_H_Y = self._empirical_entropy_from_counter(sampled_y_counts)

    @staticmethod
    def _empirical_entropy_from_counter(counter):
        total = sum(int(v) for v in counter.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in counter.values():
            p = float(count) / float(total)
            entropy -= p * math.log2(p)
        return float(entropy)

import json
import os
import random
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .ColorShape import BindingDataset


@dataclass(frozen=True)
class _GenPaths:
	root: str
	images_dir: str
	meta_path: str
	spec_path: str


_CLEVR_COLORS = [
	"gray",
	"red",
	"blue",
	"green",
	"brown",
	"purple",
	"cyan",
	"yellow",
]

_CLEVR_SHAPES = ["cube", "sphere", "cylinder"]

def _pil_to_chw_float(img: Image.Image) -> torch.Tensor:
	arr = np.asarray(img).astype(np.float32) / 255.0
	if arr.ndim == 2:
		arr = np.stack([arr, arr, arr], axis=-1)
	arr = np.transpose(arr, (2, 0, 1))
	return torch.from_numpy(arr)


class OcclusionClevrDataset(BindingDataset):
	"""OcclusionClevr dataset generated with Blender (uniform over Y).

	This rewrites the previous CLEVR reader dataset: instead of sampling existing
	CLEVR scenes (which induces CLEVR's natural, unbalanced P(Y)), this class
	*generates* new images using Blender and the assets from
	https://github.com/facebookresearch/clevr-dataset-gen.

	Key property: when cfg.dataset.distribution_mode == 'uniform', the dataset is
	uniform over all subsets Y with |Y| <= max_objects (as implemented by
	BindingDataset._sample_Y).

	Required cfg.dataset fields:
		- clevr_gen_dir: local clone of facebookresearch/clevr-dataset-gen
		- blender_path: blender executable path (or 'blender' if on PATH)
		- gen_root: cache directory for generated images + labels

	Outputs match other binding datasets:
		(img, Y, F, H_Y_given_F)
	"""

	def __init__(self, cfg, seed: int = 0, num_samples: Optional[int] = None, transform=None, output_dir=None, split: str = "train"):
		self.cfg = cfg

		# ----------------------------
		# feature space (same logic: choose first N)
		# ----------------------------
		n_colors = int(cfg.dataset.get("n_colors", 2))
		n_shapes = int(cfg.dataset.get("n_shapes", 2))
		colors = list(cfg.dataset.get("colors", _CLEVR_COLORS))[:n_colors]
		shapes = list(cfg.dataset.get("shapes", _CLEVR_SHAPES))[:n_shapes]

		super().__init__(
			feature_groups=[colors, shapes],
			distribution_mode=cfg.dataset.get("distribution_mode", "uniform"),
			max_objects=cfg.dataset.get("max_objects", None),
			transform=transform,
		)

		# inverse map (color_idx, shape_idx) -> conj_id
		self._conj_lookup: Dict[Tuple[int, int], int] = {
			(ci, si): k for k, (ci, si) in enumerate(self.conjunction_group_indices)
		}

		# ----------------------------
		# randomness
		# ----------------------------
		self.seed = seed if seed is not None else 0
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)

		# ----------------------------
		# Generation config
		# ----------------------------
		self.split = split
		self.image_size = int(cfg.dataset.get("image_size", 224))
		# Render resolution inside Blender (can be higher than image_size for clarity).
		# If not set, we render at image_size.
		self.render_size = int(cfg.dataset.get("render_size", self.image_size))
		self.render_num_samples = int(cfg.dataset.get("render_num_samples", 64))
		self.use_gpu = int(cfg.dataset.get("use_gpu", 0))
		self.gpu_backend = str(cfg.dataset.get("gpu_backend", "CUDA")).upper()
		if self.use_gpu == 1 and self.gpu_backend != "CUDA":
			raise ValueError(f"Only gpu_backend='CUDA' is supported (got {self.gpu_backend!r})")
		self.gpu_device_index = cfg.dataset.get("gpu_device_index", None)
		self.gpu_device_name = cfg.dataset.get("gpu_device_name", None)
		self.min_dist = float(cfg.dataset.get("min_dist", 0.25))
		self.margin = float(cfg.dataset.get("margin", 0.4))
		self.max_retries = int(cfg.dataset.get("max_retries", 50))
		self.light_jitter = {
			"key": float(cfg.dataset.get("key_light_jitter", 1.0)),
			"fill": float(cfg.dataset.get("fill_light_jitter", 1.0)),
			"back": float(cfg.dataset.get("back_light_jitter", 1.0)),
			"camera": float(cfg.dataset.get("camera_jitter", 0.5)),
		}
		self.camera_elevation = cfg.dataset.get("camera_elevation", None)
		if self.camera_elevation is not None:
			self.camera_elevation = float(self.camera_elevation)
		self.camera_pitch_deg = cfg.dataset.get("camera_pitch_deg", None)
		if self.camera_pitch_deg is not None:
			self.camera_pitch_deg = float(self.camera_pitch_deg)

		self.clevr_gen_dir = str(cfg.dataset.get("clevr_gen_dir", ""))
		if not self.clevr_gen_dir:
			raise ValueError(
				"cfg.dataset.clevr_gen_dir must point to a local clone of facebookresearch/clevr-dataset-gen"
			)
		if not os.path.isdir(self.clevr_gen_dir):
			raise FileNotFoundError(f"clevr_gen_dir not found: {self.clevr_gen_dir}")

		self.blender_path = str(cfg.dataset.get("blender_path", "blender"))
		self.gen_root = str(cfg.dataset.get("gen_root", os.path.join(os.path.dirname(__file__), "_data", "CLEVRGEN")))

		self.num_samples = num_samples
		n_to_generate = int(num_samples) if num_samples is not None else 1000
		self.generate_dataset(n_to_generate)

	def _cache_paths(self, n_samples: int) -> _GenPaths:
		colors_key = "-".join(self.feature_groups[0])
		shapes_key = "-".join(self.feature_groups[1])
		max_obj = int(self.max_objects) if self.max_objects is not None else self.n_conjunctions
		seed = int(self.seed)
		run_id = (
			f"{self.split}_N{int(n_samples)}_seed{seed}_M{max_obj}_C{colors_key}_S{shapes_key}"
			f"_render{int(self.render_size)}_out{int(self.image_size)}"
		)
		if self.camera_elevation is not None:
			run_id += f"_camz{self.camera_elevation:g}"
		if self.camera_pitch_deg is not None:
			run_id += f"_campitch{self.camera_pitch_deg:g}"
		root = os.path.join(self.gen_root, run_id)
		images_dir = os.path.join(root, "images")
		meta_path = os.path.join(root, "meta.pt")
		spec_path = os.path.join(root, "spec.jsonl")
		return _GenPaths(root=root, images_dir=images_dir, meta_path=meta_path, spec_path=spec_path)

	def _find_missing_images(self, image_filenames: List[str], images_dir: str) -> List[str]:
		missing: List[str] = []
		for fn in image_filenames:
			path = os.path.join(images_dir, str(fn))
			if not os.path.exists(path):
				missing.append(str(fn))
		return missing

	def _write_spec_subset(
		self,
		*,
		spec_path: str,
		out_spec_path: str,
		keep_filenames: List[str],
	) -> int:
		keep = set(str(x) for x in keep_filenames)
		count = 0
		with open(spec_path, "r") as src, open(out_spec_path, "w") as dst:
			for line in src:
				try:
					obj = json.loads(line)
					fn = str(obj.get("image_filename", ""))
				except Exception:
					continue
				if fn in keep:
					dst.write(line.rstrip("\n") + "\n")
					count += 1
		return count

	def _repair_missing_images_from_spec(
		self,
		*,
		spec_path: str,
		images_dir: str,
		missing_filenames: List[str],
		max_passes: int,
	) -> List[str]:
		"""Attempt to render a subset of missing images using the existing spec.jsonl."""
		missing = list(dict.fromkeys(str(x) for x in missing_filenames))
		if not missing:
			return []
		if not os.path.exists(spec_path):
			return missing

		max_passes = max(1, int(max_passes))
		base_dir = os.path.dirname(spec_path)
		for attempt in range(max_passes):
			out_spec = os.path.join(base_dir, f"spec.missing.pass{attempt+1}.jsonl")
			n_written = self._write_spec_subset(spec_path=spec_path, out_spec_path=out_spec, keep_filenames=missing)
			if n_written == 0:
				# Spec file doesn't contain the missing filenames; nothing to do.
				break
			self._render_with_blender(out_spec, images_dir, scenes_dir=None)
			missing = self._find_missing_images(missing, images_dir)
			if not missing:
				break
		return missing

	def _load_image(self, image_filename: str, images_dir: str) -> torch.Tensor:
		path = os.path.join(images_dir, image_filename)
		if not os.path.exists(path):
			raise FileNotFoundError(f"Generated image not found: {path}")
		img = Image.open(path).convert("RGB")
		if self.image_size is not None and self.image_size > 0:
			img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
		return _pil_to_chw_float(img)

	def _render_with_blender(self, spec_path: str, images_dir: str, scenes_dir: Optional[str]) -> None:
		render_script = os.path.join(os.path.dirname(__file__), "script", "occlusionclevr_blender_generation_script.py")
		if not os.path.exists(render_script):
			raise FileNotFoundError(f"Missing blender render script: {render_script}")

		blender_exe = shutil.which(self.blender_path) if os.path.sep not in self.blender_path else self.blender_path
		if blender_exe is None or not os.path.exists(blender_exe):
			raise FileNotFoundError(
				f"Blender executable not found: {self.blender_path}. "
				"Set cfg.dataset.blender_path to your blender binary."
			)

		# ------------------------------------------------------------
		# Multi-GPU data-parallel mode: run multiple Blender processes,
		# each pinned to one GPU and rendering a shard of spec.jsonl.
		# This avoids Cycles' single-frame multi-GPU synchronization.
		# ------------------------------------------------------------
		gpu_device_indices = self.cfg.dataset.get("gpu_device_indices", None)
		if int(self.use_gpu) == 1 and gpu_device_indices is not None:
			# Accept either a YAML list or a comma-separated string.
			if isinstance(gpu_device_indices, str):
				gpu_device_indices = [x.strip() for x in gpu_device_indices.split(",") if x.strip() != ""]
			try:
				gpu_list = [int(x) for x in list(gpu_device_indices)]
			except Exception as e:
				raise ValueError(f"cfg.dataset.gpu_device_indices must be a list of ints (got {gpu_device_indices!r})") from e
			gpu_list = [g for g in gpu_list if g >= 0]
			if len(gpu_list) >= 2:
				self._render_with_blender_multi(
					blender_exe=blender_exe,
					render_script=render_script,
					spec_path=spec_path,
					images_dir=images_dir,
					scenes_dir=scenes_dir,
					gpu_list=gpu_list,
				)
				return

		cmd = [
			blender_exe,
			"--background",
			"--python",
			render_script,
			"--",
			"--clevr_gen_dir",
			self.clevr_gen_dir,
			"--spec_jsonl",
			spec_path,
			"--output_image_dir",
			images_dir,
			"--width",
			str(int(self.render_size)),
			"--height",
			str(int(self.render_size)),
			"--render_num_samples",
			str(int(self.render_num_samples)),
			"--use_gpu",
			str(int(self.use_gpu)),
			"--gpu_backend",
			str(self.gpu_backend),
		]
		# Force single-GPU rendering to avoid multi-GPU overhead.
		if self.gpu_device_name is not None and str(self.gpu_device_name).strip() != "":
			cmd += ["--gpu_device_name", str(self.gpu_device_name)]
		elif self.gpu_device_index is not None:
			cmd += ["--gpu_device_index", str(int(self.gpu_device_index))]

		cmd += [
			"--min_dist",
			str(float(self.min_dist)),
			"--margin",
			str(float(self.margin)),
			"--max_retries",
			str(int(self.max_retries)),
			"--key_light_jitter",
			str(float(self.light_jitter["key"])),
			"--fill_light_jitter",
			str(float(self.light_jitter["fill"])),
			"--back_light_jitter",
			str(float(self.light_jitter["back"])),
			"--camera_jitter",
			str(float(self.light_jitter["camera"])),
		]
		if self.camera_elevation is not None:
			cmd += ["--camera_elevation", str(float(self.camera_elevation))]
		if self.camera_pitch_deg is not None:
			cmd += ["--camera_pitch_deg", str(float(self.camera_pitch_deg))]
		if scenes_dir is not None:
			cmd += ["--output_scene_dir", scenes_dir]

		show_tqdm = bool(self.cfg.dataset.get("show_tqdm", True))
		try:
			from tqdm.auto import tqdm  # type: ignore
		except Exception:
			tqdm = None  # type: ignore

		# Count how many scenes we expect (for a progress bar).
		total = None
		try:
			with open(spec_path, "r") as f:
				total = sum(1 for _ in f)
		except Exception:
			total = None

		# Stream Blender output so we can show progress in the parent process.
		env = os.environ.copy()
		if int(self.use_gpu) == 1 and self.gpu_device_name is None and self.gpu_device_index is not None:
			# Constrain what Blender sees as "device 0" (belt-and-suspenders with Blender-side device selection).
			env["CUDA_VISIBLE_DEVICES"] = str(int(self.gpu_device_index))

		proc = subprocess.Popen(
			cmd,
			env=env,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)

		bar = None
		if show_tqdm and tqdm is not None and total is not None:
			bar = tqdm(total=total, desc="Rendering CLEVRGEN", unit="img")
		current = 0
		progress_prefix = "__CLEVRGEN_PROGRESS__"
		assert proc.stdout is not None
		for line in proc.stdout:
			# Parse progress lines emitted by the Blender script.
			if line.startswith(progress_prefix):
				parts = line.strip().split()
				if len(parts) >= 3:
					try:
						now = int(parts[1])
						if bar is not None and now > current:
							bar.update(now - current)
						current = max(current, now)
					except Exception:
						pass
				continue
			# Otherwise, forward Blender logs.
			# When tqdm is enabled, keep output minimal but still show important lines.
			if not show_tqdm:
				print(line, end="")
			else:
				ls = line.lstrip()
				if ls.startswith("[info]") or ls.startswith("[warn]") or ls.startswith("ERROR"):
					print(line, end="")

		ret = proc.wait()
		if bar is not None:
			# Ensure bar completes.
			if total is not None and current < total:
				bar.update(total - current)
			bar.close()
		if ret != 0:
			raise subprocess.CalledProcessError(ret, cmd)

	def _render_with_blender_multi(
		self,
		*,
		blender_exe: str,
		render_script: str,
		spec_path: str,
		images_dir: str,
		scenes_dir: Optional[str],
		gpu_list: List[int],
	) -> None:
		show_tqdm = bool(self.cfg.dataset.get("show_tqdm", True))
		try:
			from tqdm.auto import tqdm  # type: ignore
		except Exception:
			tqdm = None  # type: ignore

		with open(spec_path, "r") as f:
			lines = f.readlines()
		total = len(lines)
		if total == 0:
			return

		# Number of workers can be capped via cfg.dataset.render_workers.
		n_workers_cfg = self.cfg.dataset.get("render_workers", None)
		if n_workers_cfg is not None:
			try:
				n_workers = max(1, int(n_workers_cfg))
			except Exception:
				n_workers = len(gpu_list)
			gpu_list = gpu_list[:n_workers]

		os.makedirs(images_dir, exist_ok=True)
		parts_dir = os.path.join(os.path.dirname(spec_path), "spec_parts")
		os.makedirs(parts_dir, exist_ok=True)

		# Shard specs contiguously to preserve filename ordering.
		part_paths: List[str] = []
		part_sizes: List[int] = []
		for wi in range(len(gpu_list)):
			start = (total * wi) // len(gpu_list)
			end = (total * (wi + 1)) // len(gpu_list)
			part = lines[start:end]
			part_path = os.path.join(parts_dir, f"spec.part{wi}.jsonl")
			with open(part_path, "w") as f:
				f.writelines(part)
			part_paths.append(part_path)
			part_sizes.append(len(part))

		bar = None
		if show_tqdm and tqdm is not None:
			bar = tqdm(total=total, desc=f"Rendering CLEVRGEN ({len(gpu_list)} GPUs)", unit="img")

		progress_prefix = "__CLEVRGEN_PROGRESS__"
		lock = threading.Lock()
		threads: List[threading.Thread] = []
		procs: List[subprocess.Popen] = []
		errors: List[Tuple[int, int]] = []  # (worker_id, returncode)

		def _worker_reader(proc: subprocess.Popen, worker_id: int) -> None:
			current = 0
			assert proc.stdout is not None
			for line in proc.stdout:
				if line.startswith(progress_prefix):
					parts = line.strip().split()
					if len(parts) >= 2:
						try:
							now = int(parts[1])
							delta = max(0, now - current)
							current = max(current, now)
							if bar is not None and delta > 0:
								with lock:
									bar.update(delta)
						except Exception:
							pass
					continue
				if not show_tqdm:
					print(f"[w{worker_id}] {line}", end="")
				else:
					ls = line.lstrip()
					if ls.startswith("[info]") or ls.startswith("[warn]") or ls.startswith("ERROR"):
						print(f"[w{worker_id}] {line}", end="")

		for wi, gpu_id in enumerate(gpu_list):
			if part_sizes[wi] == 0:
				continue
			cmd = [
				blender_exe,
				"--background",
				"--python",
				render_script,
				"--",
				"--clevr_gen_dir",
				self.clevr_gen_dir,
				"--spec_jsonl",
				part_paths[wi],
				"--output_image_dir",
				images_dir,
				"--width",
				str(int(self.render_size)),
				"--height",
				str(int(self.render_size)),
				"--render_num_samples",
				str(int(self.render_num_samples)),
				"--use_gpu",
				"1",
				"--gpu_backend",
				str(self.gpu_backend),
				"--gpu_device_index",
				"0",
				"--min_dist",
				str(float(self.min_dist)),
				"--margin",
				str(float(self.margin)),
				"--max_retries",
				str(int(self.max_retries)),
				"--key_light_jitter",
				str(float(self.light_jitter["key"])),
				"--fill_light_jitter",
				str(float(self.light_jitter["fill"])),
				"--back_light_jitter",
				str(float(self.light_jitter["back"])),
				"--camera_jitter",
				str(float(self.light_jitter["camera"])),
			]
			if self.camera_elevation is not None:
				cmd += ["--camera_elevation", str(float(self.camera_elevation))]
			if self.camera_pitch_deg is not None:
				cmd += ["--camera_pitch_deg", str(float(self.camera_pitch_deg))]
			if scenes_dir is not None:
				cmd += ["--output_scene_dir", scenes_dir]

			env = os.environ.copy()
			env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
			env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

			proc = subprocess.Popen(
				cmd,
				env=env,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
			)
			procs.append(proc)
			t = threading.Thread(target=_worker_reader, args=(proc, wi), daemon=True)
			threads.append(t)
			t.start()

		# Wait for all processes.
		for wi, proc in enumerate(procs):
			ret = proc.wait()
			if ret != 0:
				errors.append((wi, ret))

		for t in threads:
			t.join(timeout=1.0)

		if bar is not None:
			bar.close()

		if errors:
			# Surface the first error, but include all return codes.
			msg = ", ".join([f"w{wi}: rc={rc}" for wi, rc in errors])
			raise RuntimeError(f"One or more Blender workers failed: {msg}")

	def generate_dataset(self, n_samples: int):
		n_samples = int(n_samples)
		if n_samples <= 0:
			raise ValueError("n_samples must be positive")

		missing_policy = str(self.cfg.dataset.get("missing_policy", "raise")).lower().strip()
		repair_passes = int(self.cfg.dataset.get("repair_missing_passes", 5))
		repair_enabled = bool(self.cfg.dataset.get("repair_missing", True))
		use_cache_only = bool(self.cfg.dataset.get("use_cache_only", False))
		render_missing_only = bool(self.cfg.dataset.get("render_missing_only", True))

		paths = self._cache_paths(n_samples)
		os.makedirs(paths.images_dir, exist_ok=True)

		if use_cache_only and not os.path.exists(paths.meta_path):
			raise FileNotFoundError(
				f"use_cache_only=true but cache meta not found: {paths.meta_path}. "
				"Either set use_cache_only=false to allow generation, or generate the cache first by running once with use_cache_only=false."
			)

		if os.path.exists(paths.meta_path):
			payload = torch.load(paths.meta_path, map_location="cpu")
			cached_images_dir = str(payload.get("images_dir", paths.images_dir))
			cached_filenames = list(payload.get("image_filenames", []))[:n_samples]
			missing = self._find_missing_images([str(fn) for fn in cached_filenames], cached_images_dir)
			if len(missing) == 0 and len(cached_filenames) == n_samples:
				self.image_filenames = cached_filenames
				self.Y = payload["Y"][:n_samples].contiguous()
				self.F = payload["F"][:n_samples].contiguous()
				self.H_Y_given_F = payload["H"][:n_samples].contiguous()
				self.mean_H_Y_given_F = float(self.H_Y_given_F.mean().item())
				self._images_dir = cached_images_dir
				return

			# Cache exists but has missing images. Try to repair using the saved spec.jsonl.
			if missing and repair_enabled and os.path.exists(paths.spec_path):
				print(f"[warn] CLEVRGEN cache incomplete: {len(missing)}/{len(cached_filenames)} images missing. Attempting repair...")
				missing = self._repair_missing_images_from_spec(
					spec_path=paths.spec_path,
					images_dir=cached_images_dir,
					missing_filenames=missing,
					max_passes=repair_passes,
				)
				if not missing and len(cached_filenames) == n_samples:
					self.image_filenames = cached_filenames
					self.Y = payload["Y"][:n_samples].contiguous()
					self.F = payload["F"][:n_samples].contiguous()
					self.H_Y_given_F = payload["H"][:n_samples].contiguous()
					self.mean_H_Y_given_F = float(self.H_Y_given_F.mean().item())
					self._images_dir = cached_images_dir
					return

			if missing and missing_policy in {"drop", "filter"}:
				print(f"[warn] Dropping {len(missing)} missing images from cached dataset.")
				missing_set = set(str(x) for x in missing)
				keep_idx = [i for i, fn in enumerate(cached_filenames) if str(fn) not in missing_set]
				self.image_filenames = [cached_filenames[i] for i in keep_idx]
				self.Y = payload["Y"][keep_idx].contiguous()
				self.F = payload["F"][keep_idx].contiguous()
				self.H_Y_given_F = payload["H"][keep_idx].contiguous()
				self.mean_H_Y_given_F = float(self.H_Y_given_F.mean().item()) if len(keep_idx) > 0 else 0.0
				self._images_dir = cached_images_dir
				# Persist the filtered cache so subsequent runs are consistent.
				torch.save(
					{
						"image_filenames": self.image_filenames,
						"Y": self.Y,
						"F": self.F,
						"H": self.H_Y_given_F,
						"images_dir": self._images_dir,
					},
					paths.meta_path,
				)
				return

			if missing:
				raise FileNotFoundError(
					f"CLEVRGEN cache incomplete: {len(missing)}/{len(cached_filenames)} images missing under {cached_images_dir}. "
					"Enable repair (cfg.dataset.repair_missing=true) and/or increase cfg.dataset.repair_missing_passes, "
					"or set cfg.dataset.missing_policy='drop' to filter missing samples."
				)

			# Cache exists but doesn't match requested size. Regenerate.
			try:
				os.remove(paths.meta_path)
			except OSError:
				pass

		seed = int(self.seed)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

		image_filenames: List[str] = []
		Ys: List[torch.Tensor] = []
		Fs: List[torch.Tensor] = []
		Hs: List[torch.Tensor] = []
		spec_lines: List[str] = []

		show_tqdm = bool(self.cfg.dataset.get("show_tqdm", True))
		try:
			from tqdm.auto import tqdm  # type: ignore
		except Exception:
			tqdm = None  # type: ignore

		itr = range(n_samples)
		if show_tqdm and tqdm is not None:
			itr = tqdm(itr, desc="Sampling specs", unit="img")

		for i in itr:
			y = self._sample_Y()
			f = self.Y_to_F(y)
			h = self.get_H_Y_F(f)

			active = torch.nonzero(y > 0, as_tuple=False).view(-1).tolist()
			objects: List[Dict[str, Any]] = []
			for conj_id in active:
				ci, si = self.conjunction_group_indices[int(conj_id)]
				objects.append(
					{
						"color": self.feature_groups[0][int(ci)],
						"shape": self.feature_groups[1][int(si)],
						"size": str(self.cfg.dataset.get("default_size", "small")),
						"material": str(self.cfg.dataset.get("default_material", "rubber")),
					}
				)

			image_filename = f"CLEVRGEN_{self.split}_{i:06d}.png"
			image_filenames.append(image_filename)
			Ys.append(y)
			Fs.append(f)
			Hs.append(h)

			spec_lines.append(
				json.dumps(
					{
						"split": self.split,
						"image_index": i,
						"image_filename": image_filename,
						"seed": seed + i,
						"objects": objects,
					},
					sort_keys=True,
				)
			)

		with open(paths.spec_path, "w") as sf:
			sf.write("\n".join(spec_lines) + "\n")

		if render_missing_only:
			missing_before = self._find_missing_images(image_filenames, paths.images_dir)
			if missing_before:
				spec_to_render = os.path.join(paths.root, "spec.to_render.jsonl")
				n_written = self._write_spec_subset(
					spec_path=paths.spec_path,
					out_spec_path=spec_to_render,
					keep_filenames=missing_before,
				)
				if n_written > 0:
					self._render_with_blender(spec_to_render, paths.images_dir, scenes_dir=None)
			else:
				print("[info] All requested images already exist; skipping Blender render.")
		else:
			self._render_with_blender(paths.spec_path, paths.images_dir, scenes_dir=None)

		# Validate render completeness; repair missing frames before saving meta.
		missing = self._find_missing_images(image_filenames, paths.images_dir)
		if missing and repair_enabled:
			print(f"[warn] Render produced {len(missing)} missing images. Attempting repair...")
			missing = self._repair_missing_images_from_spec(
				spec_path=paths.spec_path,
				images_dir=paths.images_dir,
				missing_filenames=missing,
				max_passes=repair_passes,
			)

		if missing:
			if missing_policy in {"drop", "filter"}:
				print(f"[warn] Dropping {len(missing)} missing images from newly generated dataset.")
				missing_set = set(str(x) for x in missing)
				keep_idx = [i for i, fn in enumerate(image_filenames) if str(fn) not in missing_set]
				image_filenames = [image_filenames[i] for i in keep_idx]
				Ys = [Ys[i] for i in keep_idx]
				Fs = [Fs[i] for i in keep_idx]
				Hs = [Hs[i] for i in keep_idx]
			else:
				raise FileNotFoundError(
					f"CLEVRGEN render incomplete: {len(missing)}/{len(image_filenames)} images missing under {paths.images_dir}. "
					"Set cfg.dataset.missing_policy='drop' to filter missing samples, or increase cfg.dataset.max_retries / relax placement constraints."
				)

		self.image_filenames = image_filenames
		self.Y = torch.stack(Ys, 0).contiguous()
		self.F = torch.stack(Fs, 0).contiguous()
		self.H_Y_given_F = torch.stack(Hs, 0).contiguous()
		self.mean_H_Y_given_F = float(self.H_Y_given_F.mean().item())
		self._images_dir = paths.images_dir

		torch.save(
			{
				"image_filenames": self.image_filenames,
				"Y": self.Y,
				"F": self.F,
				"H": self.H_Y_given_F,
				"images_dir": self._images_dir,
			},
			paths.meta_path,
		)

	def __getitem__(self, idx: int):
		img = self._load_image(self.image_filenames[int(idx)], images_dir=getattr(self, "_images_dir", ""))
		if getattr(self, "transform", None) is not None:
			img = self.transform(img)
		return img, self.Y[idx], self.F[idx], self.H_Y_given_F[idx]


if __name__ == "__main__":
	# Smoke test (will invoke Blender). Set env vars:
	#   export CLEVR_GEN_DIR=/path/to/clevr-dataset-gen
	#   export BLENDER_PATH=/path/to/blender
	try:
		from omegaconf import OmegaConf
	except Exception as e:
		raise RuntimeError("Install 'omegaconf' to run this smoke test.") from e
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

	cfg = OmegaConf.create(
		{
			"dataset": {
				"name": "occlusionclevr",
				"n_colors": 4,
				"n_shapes": 3,
				"image_size": 448,
				"render_size": 448,
				"max_objects": 6,
				"seed": 10127,
				"clevr_gen_dir": os.environ.get("CLEVR_GEN_DIR", os.path.join(repo_root, "clevr-dataset-gen")),
				"blender_path": os.environ.get("BLENDER_PATH", "blender"),
				"gen_root": os.environ.get(
					"OCCLUSION_CLEVR_GEN_ROOT",
					os.environ.get("CLEVR_GEN_ROOT", os.path.join(repo_root, "data", "generated", "test_occlusionclevr")),
				),
				"render_num_samples": 64,
				"use_gpu": 1,
				"gpu_backend": "CUDA",
				# "gpu_device_index": 0,
				"gpu_device_indices": [0, 1, 2, 3],
				"camera_elevation": 3.2,
				"camera_pitch_deg": 32,
			},
			"model": {"load_from": "timm", "name": "vit_base_patch16_224", "output_mode": "cls"},
		}
	)

	ds = OcclusionClevrDataset(cfg, seed=int(cfg.dataset.seed), num_samples=100)
	print(f"Generated {len(ds)} samples (cached under cfg.dataset.gen_root).")

	def _decode_Y(ds_obj: OcclusionClevrDataset, y_vec: torch.Tensor) -> List[Tuple[str, str]]:
		active = torch.nonzero(y_vec > 0, as_tuple=False).view(-1).tolist()
		pairs: List[Tuple[str, str]] = []
		for conj_id in active:
			ci, si = ds_obj.conjunction_group_indices[int(conj_id)]
			pairs.append((ds_obj.feature_groups[0][int(ci)], ds_obj.feature_groups[1][int(si)]))
		return pairs

	def _decode_F(ds_obj: OcclusionClevrDataset, f_vec: torch.Tensor) -> Dict[str, List[str]]:
		out: Dict[str, List[str]] = {}
		for g, group_name in enumerate(["colors", "shapes"]):
			off = ds_obj.group_offsets[g]
			sz = ds_obj.group_sizes[g]
			present = []
			for j in range(sz):
				if int(f_vec[off + j].item()) == 1:
					present.append(ds_obj.feature_groups[g][j])
			out[group_name] = present
		return out

	for i in range(len(ds)):
		img, Y, F, H = ds[i]
		fname = ds.image_filenames[i] if hasattr(ds, "image_filenames") else "(unknown)"
		y_pairs = _decode_Y(ds, Y)
		f_dec = _decode_F(ds, F)
		print(f"[{i}] {fname} |Y|={int(Y.sum().item())} H(Y|F)={float(H.item()):.3f} bits")
		print(f"  Y (decoded): {y_pairs}")
		print(f"  F (decoded): colors={f_dec['colors']} shapes={f_dec['shapes']}")
		print(f"  Y (raw): {Y.tolist()}")
		print(f"  F (raw): {F.tolist()}")

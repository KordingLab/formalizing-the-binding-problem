import os
import logging
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from probes import get_probe
from activation import ModelWrapper
import timm
import numpy as np
import math
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.train_probe_Y = bool(getattr(cfg.trainer, "train_probe_Y", True))
        self.train_probe_F = bool(getattr(cfg.trainer, "train_probe_F", True))
        if not self.train_probe_Y and not self.train_probe_F:
            raise ValueError("At least one probe must be enabled (trainer.train_probe_Y or trainer.train_probe_F).")
        set_random_seed(cfg.seed)

        # Build backbone first so we can infer/verify expected image size.
        self.device = self.cfg.device
        self.model = ModelWrapper(cfg).to(self.device)

        def _infer_model_image_size() -> int:
            # Prefer explicit config override if present.
            if hasattr(cfg, "model") and getattr(cfg.model, "image_size", None) is not None:
                try:
                    return int(cfg.model.image_size)
                except Exception:
                    pass

            if cfg.model.load_from == "timm":
                input_size = getattr(getattr(self.model, "backbone", None), "default_cfg", {}).get("input_size", None)
                if isinstance(input_size, (tuple, list)) and len(input_size) == 3:
                    return int(input_size[1])
            if cfg.model.load_from == "huggingface":
                ip = getattr(self.model, "image_processor", None)
                # Common HF processors expose .size or .crop_size
                for attr in ("size", "crop_size"):
                    val = getattr(ip, attr, None)
                    if isinstance(val, dict):
                        h = val.get("height", None) or val.get("shortest_edge", None)
                        if h is not None:
                            return int(h)
                    if isinstance(val, (tuple, list)) and len(val) >= 1:
                        return int(val[0])

            return int(cfg.dataset.image_size)

        target_size = _infer_model_image_size()
        self.target_size = int(target_size)
        self.pin_memory = bool(getattr(cfg.trainer, "pin_memory", False)) and str(self.device).startswith("cuda")
        self.use_cached_activations = True
        self._activation_cache_datasets_by_path = {}

        def _resize_to_model(img: torch.Tensor) -> torch.Tensor:
            # img: [C,H,W] float
            if not torch.is_tensor(img):
                raise TypeError(f"Expected torch.Tensor image, got {type(img)}")
            if img.ndim != 3:
                raise ValueError(f"Expected image tensor [C,H,W], got shape={tuple(img.shape)}")
            _, h, w = img.shape
            if int(h) == int(target_size) and int(w) == int(target_size):
                return img
            out = F.interpolate(img.unsqueeze(0), size=(int(target_size), int(target_size)), mode="bilinear", align_corners=False)
            return out.squeeze(0)

        # If output_mode=='all', z_dim depends on cfg.dataset.image_size; enforce consistency.
        if getattr(cfg.model, "output_mode", "cls") == "all" and int(cfg.dataset.image_size) != int(target_size):
            raise ValueError(
                f"Model output_mode='all' requires cfg.dataset.image_size==model input size. "
                f"Got dataset.image_size={int(cfg.dataset.image_size)} vs model={int(target_size)}."
            )

        DatasetCls = get_dataset(cfg)

        natural_train_dataset = DatasetCls(cfg, seed=cfg.dataset.seed, num_samples=cfg.dataset.num_train_samples * 0.8, transform=_resize_to_model, output_dir=self.output_dir, split='train')
        val_dataset = DatasetCls(cfg, seed=cfg.dataset.seed + 1, num_samples=cfg.dataset.num_train_samples * 0.2, transform=_resize_to_model, output_dir=self.output_dir, split='val')
        test_dataset = DatasetCls(cfg, seed=cfg.dataset.seed + 2, num_samples=cfg.dataset.num_test_samples, transform=_resize_to_model, output_dir=self.output_dir, split='test')
        train_dataset = natural_train_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=cfg.trainer.num_workers, pin_memory=self.pin_memory)
        self.eval_train_dataloader = DataLoader(natural_train_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=cfg.trainer.num_workers, pin_memory=self.pin_memory)
        self.val_dataloader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=cfg.trainer.num_workers, pin_memory=self.pin_memory)


        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=cfg.trainer.num_workers, pin_memory=self.pin_memory)

        self.y_dim = train_dataset.n_conjunctions
        self.f_dim = train_dataset.n_features
        self.z_dim = self.model.z_dim

        attention_probe_types = {
            "attention_quadratic",
        }
        if (
            (self.train_probe_Y and cfg.probes.probe_Y.type in attention_probe_types)
            or (self.train_probe_F and cfg.probes.probe_F.type in attention_probe_types)
        ) and cfg.model.output_mode != "all":
            raise ValueError(
                "Token-based attention probes require model.output_mode='all' "
                f"(got '{cfg.model.output_mode}')."
            )

        self._cache_all_dataloader_activations()

        # get_probe(...) accepts common args plus optional architecture-specific args.
        self.probe_Y = None
        if self.train_probe_Y:
            self.probe_Y = get_probe(
                cfg.probes.probe_Y.type,
                self.z_dim,
                self.y_dim,
                self.y_dim,
                rank=cfg.probes.probe_Y.get("rank", None),
                feature_dims=train_dataset.group_sizes,
                token_dim=self.model.token_dim,
                attend_cls=False,
                dnn_hidden_dim=cfg.probes.probe_Y.get("dnn_hidden_dim", None),
                dnn_num_layers=cfg.probes.probe_Y.get("dnn_num_layers", 3),
                dnn_dropout=cfg.probes.probe_Y.get("dnn_dropout", 0.0),
                use_conditioned_query=cfg.probes.probe_Y.get("use_conditioned_query", True),
            ).to(self.device)
            logger.info("%s", self.probe_Y)
            logger.info(f"ProbeY parameters: {count_trainable_params(self.probe_Y):,}")
            print_parameter_shapes(self.probe_Y, "ProbeY")
        else:
            logger.info("ProbeY disabled (trainer.train_probe_Y=false).")

        self.probe_F = None
        if self.train_probe_F:
            self.probe_F = get_probe(
                cfg.probes.probe_F.type,
                self.z_dim,
                self.f_dim,
                self.f_dim,
                rank=cfg.probes.probe_F.get("rank", None),
                token_dim=self.model.token_dim,
                attend_cls=False,
                dnn_hidden_dim=cfg.probes.probe_F.get("dnn_hidden_dim", None),
                dnn_num_layers=cfg.probes.probe_F.get("dnn_num_layers", 3),
                dnn_dropout=cfg.probes.probe_F.get("dnn_dropout", 0.0),
                use_conditioned_query=cfg.probes.probe_F.get("use_conditioned_query", True),
            ).to(self.device)
            logger.info("%s", self.probe_F)
            logger.info(f"ProbeF parameters: {count_trainable_params(self.probe_F):,}")
            print_parameter_shapes(self.probe_F, "ProbeF")
        else:
            logger.info("ProbeF disabled (trainer.train_probe_F=false).")

        self.criterion_Y = nn.BCEWithLogitsLoss() if self.train_probe_Y else None
        self.criterion_F = nn.BCEWithLogitsLoss() if self.train_probe_F else None

        params = []
        if self.train_probe_Y:
            params.extend(list(self.probe_Y.parameters()))
        if self.train_probe_F:
            params.extend(list(self.probe_F.parameters()))
        if len(params) == 0:
            raise ValueError("No trainable probe parameters found. Enable at least one probe.")

        self.optimizer = optim.Adam(params, lr=self.cfg.trainer.learning_rate, weight_decay=self.cfg.trainer.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.cfg.trainer.scheduler.step_size, gamma=self.cfg.trainer.scheduler.gamma)

        self.ckpt_path = f"{self.output_dir}/checkpoint.pth"

        self.start_epoch = 0
        self.best_val = {"Y": 1e5 if self.train_probe_Y else None, "F": 1e5 if self.train_probe_F else None}
        self.best_test = {"Y": 1e5 if self.train_probe_Y else None, "F": 1e5 if self.train_probe_F else None}
        
    def train(self):
        
        if os.path.exists(self.ckpt_path):
            logger.info(f'checkpoint found at {self.ckpt_path}, skip training...')
            self.load_model(self.ckpt_path)
            eval_dict = self.eval("test")
            self._maybe_save_probe_weights_if_missing()
            return eval_dict

        init_checkpoint_path = getattr(self.cfg.trainer, "init_checkpoint_path", None)
        if init_checkpoint_path is not None:
            init_checkpoint_path = str(init_checkpoint_path)
            if init_checkpoint_path:
                logger.info(f"Initializing trainable probes from {init_checkpoint_path}")
                self.load_model(init_checkpoint_path)
            

        grad_accum_steps = int(getattr(self.cfg.trainer, "grad_accum_steps", 1))
        if grad_accum_steps < 1:
            raise ValueError(f"trainer.grad_accum_steps must be >= 1 (got {grad_accum_steps})")

        eval_splits_each_epoch = ["val", "test"]
        training_history = []

        for epoch in range(self.start_epoch, self.cfg.trainer.max_epoch):
            stat_dict = {}
            if self.train_probe_Y:
                stat_dict["train_lossY"] = 0.0
                stat_dict["train_accY"] = 0.0
            if self.train_probe_F:
                stat_dict["train_lossF"] = 0.0
                stat_dict["train_accF"] = 0.0

            self.model.eval()  # backbone frozen; keep deterministic stats
            if self.train_probe_Y:
                self.probe_Y.train()
            if self.train_probe_F:
                self.probe_F.train()

            total_num = 0
            self.optimizer.zero_grad(set_to_none=True)

            for step_idx, (inputs, Y, F, _) in enumerate(tqdm(self.train_dataloader)):
                outputs = self.train_loop(
                    inputs,
                    Y,
                    F,
                    inputs_are_z=self.use_cached_activations,
                )
                losses = []
                if self.train_probe_Y:
                    losses.append(outputs["lossY"])
                if self.train_probe_F:
                    losses.append(outputs["lossF"])
                # Scale the loss so that accumulated gradients match the non-accumulated case.
                loss = sum(losses) / float(grad_accum_steps)
                loss.backward()

                do_step = ((step_idx + 1) % grad_accum_steps == 0) or ((step_idx + 1) == len(self.train_dataloader))
                if do_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                B = inputs.shape[0]
                if self.train_probe_Y:
                    stat_dict["train_lossY"] += outputs["lossY"].item() * B * self.y_dim / math.log(2)
                    stat_dict["train_accY"] += outputs["accY"] * B
                if self.train_probe_F:
                    stat_dict["train_lossF"] += outputs["lossF"].item() * B * self.f_dim / math.log(2)
                    stat_dict["train_accF"] += outputs["accF"] * B
                total_num += B

            self.scheduler.step()
            for k in stat_dict:
                stat_dict[k] /= max(1, total_num)
            if self.train_probe_Y and self.train_probe_F:
                self._add_binding_metrics(stat_dict, "train", self.train_dataloader.dataset)

            train_parts = [f"[Epoch {epoch+1}]"]
            if self.train_probe_Y:
                train_parts.append(f'lossY: {stat_dict["train_lossY"]:.4f}')
                train_parts.append(f'accY: {stat_dict["train_accY"]*100:.2f}%')
            if self.train_probe_F:
                train_parts.append(f'lossF: {stat_dict["train_lossF"]:.4f}')
                train_parts.append(f'accF: {stat_dict["train_accF"]*100:.2f}%')
            logger.info(", ".join(train_parts))

            eval_dicts = {split: self.eval(split) for split in eval_splits_each_epoch}
            val_dict = eval_dicts.get("val")
            test_dict = eval_dicts.get("test", {})
            epoch_record = {"epoch": epoch + 1}
            epoch_record.update(stat_dict)
            for split_dict in eval_dicts.values():
                epoch_record.update(split_dict)
            training_history.append(epoch_record)

            if self.train_probe_Y and val_dict is not None and val_dict["val_lossY"] < self.best_val["Y"]:
                self.best_val["Y"] = val_dict["val_lossY"]
                self.best_test["Y"] = test_dict.get("test_lossY")
                logger.info(
                    f"New best val lossY: {self.best_val['Y']:.4f}, "
                    f"corresponding test lossY: {self.best_test['Y']}"
                )
                self.save_model(self.ckpt_path, update_probeY=True, update_probeF=False)
            if self.train_probe_F and val_dict is not None and val_dict["val_lossF"] < self.best_val["F"]:
                self.best_val["F"] = val_dict["val_lossF"]
                self.best_test["F"] = test_dict.get("test_lossF")
                logger.info(
                    f"New best val lossF: {self.best_val['F']:.4f}, "
                    f"corresponding test lossF: {self.best_test['F']}"
                )
                self.save_model(self.ckpt_path, update_probeY=False, update_probeF=True)

        self.save_model(self.ckpt_path, update_probeY=True, update_probeF=True)
        self._maybe_save_probe_weights_if_missing()
        self._maybe_plot_training_history(training_history)

    def _activation_cache_mode(self):
        mode = str(getattr(self.cfg.trainer, "activation_cache_mode", "generate")).strip().lower()
        aliases = {
            "generate": "generate",
            "regen": "generate",
            "regenerate": "generate",
            "refresh": "generate",
            "load": "load",
            "saved": "load",
            "existing": "load",
        }
        if mode not in aliases:
            raise ValueError(
                "trainer.activation_cache_mode must be 'generate' or 'load' "
                f"(got {mode!r})."
            )
        return aliases[mode]

    def _activation_cache_root(self):
        cache_dir = getattr(self.cfg.trainer, "activation_cache_dir", None)
        if cache_dir in (None, "", "null"):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            cache_dir = os.path.join(project_root, "data", "cache", "activations")
        cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _safe_cache_name(value):
        keep = []
        for ch in str(value):
            keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "-")
        return "".join(keep).strip("-") or "unknown"

    @staticmethod
    def _update_hash_tensor(hasher, name, tensor):
        tensor = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())

    @staticmethod
    def _update_hash_json(hasher, name, value):
        hasher.update(name.encode("utf-8"))
        hasher.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))

    def _dataset_content_hash(self, dataset):
        hasher = hashlib.sha1()
        self._update_hash_json(hasher, "dataset_class", dataset.__class__.__name__)
        self._update_hash_json(hasher, "len", len(dataset))
        for attr in (
            "Y",
            "F",
            "H_Y_given_F",
            "images",
            "num_objects_per_sample",
        ):
            value = getattr(dataset, attr, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                self._update_hash_tensor(hasher, attr, value)
            else:
                self._update_hash_json(hasher, attr, value)
        for attr in ("boxes_list", "image_metadata_extras"):
            value = getattr(dataset, attr, None)
            if value is not None:
                self._update_hash_json(hasher, attr, value)
        return hasher.hexdigest()

    def _all_token_z_dim(self):
        num_patches = (int(self.target_size) // int(self.model.patch_size)) ** 2
        return int((num_patches + 1) * self.model.token_dim)

    def _activation_cache_metadata(self, split_name, dataset, cache_dtype):
        dataset_attrs = {}
        for attr in (
            "split",
            "seed",
            "num_samples",
            "distribution_mode",
            "split_mode",
            "exclude_all_ones_f",
            "f_split_hash_mod",
            "f_split_test_buckets",
            "f_split_val_buckets",
            "f_split_train_count",
            "f_split_val_count",
            "f_split_test_count",
            "f_split_salt",
            "n_features",
            "n_conjunctions",
            "group_sizes",
            "feature_groups",
            "max_objects",
            "num_objects",
            "candidate_colors_per_sample",
            "candidate_shapes_per_sample",
            "image_size",
            "obj_size",
            "margin",
            "max_place_tries",
            "non_overlapping",
            "background",
        ):
            if hasattr(dataset, attr):
                dataset_attrs[attr] = getattr(dataset, attr)

        return {
            "cache_version": 2,
            "activation_format": "all_tokens_flat",
            "model": {
                "name": str(self.cfg.model.name),
                "short_name": str(getattr(self.cfg.model, "short_name", self.cfg.model.name)),
                "load_from": str(self.cfg.model.load_from),
                "target_size": int(self.target_size),
                "z_dim": int(self._all_token_z_dim()),
                "token_dim": int(self.model.token_dim),
                "patch_size": int(self.model.patch_size),
            },
            "dataset": {
                "name": str(self.cfg.dataset.name),
                "class": dataset.__class__.__name__,
                "len": int(len(dataset)),
                "attrs": dataset_attrs,
                "content_hash": self._dataset_content_hash(dataset),
            },
            "cache_dtype": str(cache_dtype).replace("torch.", ""),
        }

    @staticmethod
    def _activation_cached_dataset_attrs(source_dataset):
        attrs = {}
        for attr in (
            "mean_H_Y_given_F",
            "group_sizes",
            "n_conjunctions",
            "n_features",
            "theoretical_H_F",
            "theoretical_H_Y_given_F",
            "theoretical_H_Y",
            "H_F",
            "H_Y",
            "split_f_support_count",
            "total_f_support_count",
            "y_support_per_f_count",
            "split_y_support_count",
            "num_objects_histogram",
        ):
            if hasattr(source_dataset, attr):
                attrs[attr] = getattr(source_dataset, attr)
        return attrs

    def _activation_cache_path(self, split_name, dataset, cache_dtype):
        metadata = self._activation_cache_metadata(split_name, dataset, cache_dtype)
        digest = hashlib.sha1(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        dataset_name = self._safe_cache_name(self.cfg.dataset.name)
        model_name = self._safe_cache_name(getattr(self.cfg.model, "short_name", self.cfg.model.name))
        activation_format = self._safe_cache_name(metadata["activation_format"])
        actual_split = self._safe_cache_name(getattr(dataset, "split", split_name))
        distribution = self._safe_cache_name(getattr(dataset, "distribution_mode", "unknown"))
        filename = (
            f"{dataset_name}_{model_name}_{activation_format}_{actual_split}_"
            f"{distribution}_{len(dataset)}_{digest[:16]}.pt"
        )
        return os.path.join(self._activation_cache_root(), filename), metadata

    @staticmethod
    def _load_activation_payload(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _cached_tensor_dataset(self, payload, source_dataset):
        cached_dataset = TensorDataset(
            payload["Z"].contiguous(),
            payload["Y"].contiguous(),
            payload["F"].contiguous(),
            payload["H"].contiguous(),
        )
        attrs = dict(payload.get("attrs", {}))
        for attr, value in self._activation_cached_dataset_attrs(source_dataset).items():
            if attr not in attrs:
                attrs[attr] = value
        for attr, value in attrs.items():
            setattr(cached_dataset, attr, value)
        return cached_dataset

    def _save_activation_payload(self, path, payload):
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _payload_for_current_output_mode(self, payload):
        return {
            "metadata": payload.get("metadata", {}),
            "attrs": payload.get("attrs", {}),
            "Z": self.model.format_flattened_tokens(
                payload["Z"].contiguous(),
                output_mode=str(self.cfg.model.output_mode),
            ).contiguous(),
            "Y": payload["Y"].contiguous(),
            "F": payload["F"].contiguous(),
            "H": payload["H"].contiguous(),
        }

    def _activation_cache_loader(self, cached_dataset, shuffle):
        return DataLoader(
            cached_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.pin_memory,
        )

    def _cache_dataloader_activations(self, split_name, dataloader, shuffle):
        if dataloader is None:
            return None

        dataset = dataloader.dataset
        cache_dtype = torch.float32
        cache_mode = self._activation_cache_mode()
        cache_path, cache_metadata = self._activation_cache_path(split_name, dataset, cache_dtype)

        if cache_path in self._activation_cache_datasets_by_path:
            logger.info(f"Reusing loaded activation cache for {split_name}: {cache_path}")
            return self._activation_cache_loader(self._activation_cache_datasets_by_path[cache_path], shuffle=shuffle)

        if cache_mode == "load":
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"No saved all-token activation cache found for {split_name}: {cache_path}\n"
                    "Set trainer.activation_cache_mode=generate once to build it. The saved cache is reused "
                    "for any model.output_mode by selecting the needed activations in memory."
                )
            payload = self._load_activation_payload(cache_path)
            if payload.get("metadata") != cache_metadata:
                raise ValueError(
                    f"Saved all-token activation cache metadata does not match current run for {split_name}: {cache_path}\n"
                    "Set trainer.activation_cache_mode=generate to rebuild it for this dataset/model config."
                )
            payload = self._payload_for_current_output_mode(payload)
            cached_dataset = self._cached_tensor_dataset(payload, dataset)
            self._activation_cache_datasets_by_path[cache_path] = cached_dataset
            logger.info(
                f"Loaded {split_name} all-token activations from disk cache and selected "
                f"output_mode='{self.cfg.model.output_mode}' in memory: {cache_path}"
            )
            return self._activation_cache_loader(cached_dataset, shuffle=shuffle)

        if os.path.exists(cache_path):
            logger.info(f"Regenerating {split_name} activations and overwriting cache: {cache_path}")

        cache_loader = DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.num_workers,
            pin_memory=self.pin_memory,
        )
        z_parts, y_parts, f_parts, h_parts = [], [], [], []

        self.model.eval()
        logger.info(
            f"Caching {split_name} all-token activations: {len(dataset)} samples, "
            f"batch_size={self.cfg.trainer.batch_size}, dtype={cache_dtype}."
        )
        with torch.no_grad():
            for images, Y, F, H in tqdm(cache_loader, desc=f"cache:{split_name}"):
                images = images.to(self.device, non_blocking=self.pin_memory)
                Z = self.model.forward_all_tokens(images).detach().to(dtype=cache_dtype).cpu()
                z_parts.append(Z)
                y_parts.append(Y.cpu())
                f_parts.append(F.cpu())
                h_parts.append(H.cpu())

        payload = {
            "metadata": cache_metadata,
            "attrs": self._activation_cached_dataset_attrs(dataset),
            "Z": torch.cat(z_parts, dim=0).contiguous(),
            "Y": torch.cat(y_parts, dim=0).contiguous(),
            "F": torch.cat(f_parts, dim=0).contiguous(),
            "H": torch.cat(h_parts, dim=0).contiguous(),
        }

        self._save_activation_payload(cache_path, payload)
        logger.info(f"Saved {split_name} all-token activations to disk cache: {cache_path}")

        cached_dataset = self._cached_tensor_dataset(self._payload_for_current_output_mode(payload), dataset)
        self._activation_cache_datasets_by_path[cache_path] = cached_dataset
        return self._activation_cache_loader(cached_dataset, shuffle=shuffle)

    def _cache_all_dataloader_activations(self):
        logger.info(
            "Preparing frozen backbone activations for probe training/evaluation "
            f"(mode={self._activation_cache_mode()})."
        )
        self.train_dataloader = self._cache_dataloader_activations("train", self.train_dataloader, shuffle=True)
        self.eval_train_dataloader = self._cache_dataloader_activations("eval_train", self.eval_train_dataloader, shuffle=False)
        self.val_dataloader = self._cache_dataloader_activations("val", self.val_dataloader, shuffle=False)
        self.test_dataloader = self._cache_dataloader_activations("test", self.test_dataloader, shuffle=False)

        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()
        logger.info("Finished caching activations; moved frozen backbone off the training device.")

    def eval(self, split="val"):
        stat_dict = {}
        if self.train_probe_Y:
            stat_dict[f"{split}_lossY"] = 0.0
            stat_dict[f"{split}_accY"] = 0.0
        if self.train_probe_F:
            stat_dict[f"{split}_lossF"] = 0.0
            stat_dict[f"{split}_accF"] = 0.0
        self.model.eval()
        if self.train_probe_Y:
            self.probe_Y.eval()
        if self.train_probe_F:
            self.probe_F.eval()

        dataloaders = {
            "train": self.eval_train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader,
        }
        if split not in dataloaders or dataloaders[split] is None:
            raise ValueError(f"No dataloader available for split={split!r}.")
        dataloader = dataloaders[split]

        total_num = 0
        
        for inputs, Y, F, H in tqdm(dataloader):
            with torch.no_grad():
                outputs = self.train_loop(
                    inputs,
                    Y,
                    F,
                    inputs_are_z=self.use_cached_activations,
                )

            B = inputs.shape[0]
            if self.train_probe_Y:
                stat_dict[f"{split}_lossY"] += outputs["lossY"].item() * B * self.y_dim / math.log(2)
                stat_dict[f"{split}_accY"] += outputs["accY"] * B
            if self.train_probe_F:
                stat_dict[f"{split}_lossF"] += outputs["lossF"].item() * B * self.f_dim / math.log(2)
                stat_dict[f"{split}_accF"] += outputs["accF"] * B
            total_num += B

        for k in stat_dict:
            stat_dict[k] /= max(1, total_num)

        eval_parts = [f"[Eval:{split}]"]
        if self.train_probe_Y:
            eval_parts.append(f'lossY: {stat_dict[f"{split}_lossY"]:.4f}')
            eval_parts.append(f'accY: {stat_dict[f"{split}_accY"]*100:.2f}%')
        if self.train_probe_F:
            eval_parts.append(f'lossF: {stat_dict[f"{split}_lossF"]:.4f}')
            eval_parts.append(f'accF: {stat_dict[f"{split}_accF"]*100:.2f}%')
        if self.train_probe_Y and self.train_probe_F:
            binding_info, binding_ratio = self._add_binding_metrics(stat_dict, split, dataloader.dataset)
            eval_parts.append(f"Binding Info: {binding_info:.4f}")
            eval_parts.append(f"Binding Ratio: {binding_ratio:.4f}")
        logger.info(", ".join(eval_parts))

        return stat_dict

    def _add_binding_metrics(self, stat_dict, split, dataset):
        mean_H_Y_given_F = getattr(dataset, "mean_H_Y_given_F", None)
        if mean_H_Y_given_F is None:
            raise AttributeError(f"Dataset for split={split!r} is missing mean_H_Y_given_F.")
        mean_H_Y_given_F = float(mean_H_Y_given_F)
        loss_Y = stat_dict[f"{split}_lossY"]
        loss_F = stat_dict[f"{split}_lossF"]
        binding_info = mean_H_Y_given_F - (loss_Y - loss_F)
        binding_ratio = binding_info / mean_H_Y_given_F if mean_H_Y_given_F > 0 else 0.0
        stat_dict[f"{split}_binding_info"] = binding_info
        stat_dict[f"{split}_binding_ratio"] = binding_ratio
        return binding_info, binding_ratio

    def _maybe_plot_training_history(self, history):
        enabled = bool(getattr(self.cfg.trainer, "plot_training_curves", True))
        if not enabled or len(history) == 0:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Could not import matplotlib; skipping training-curve plot: {exc}")
            return None

        output_path = getattr(self.cfg.trainer, "training_curves_path", None)
        if output_path in (None, "", "null"):
            output_path = os.path.join(self.output_dir, "training_curves.png")
        output_path = os.path.abspath(os.path.expanduser(str(output_path)))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        history_json_path = os.path.splitext(output_path)[0] + ".json"
        try:
            with open(history_json_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save training history JSON: {exc}")

        epochs = [int(row["epoch"]) for row in history]
        splits = ("train", "val", "test")
        split_colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
        metric_styles = {"Y": "-", "F": "--"}

        def _values(key, scale=1.0):
            vals = []
            for row in history:
                value = row.get(key, float("nan"))
                if value is None:
                    value = float("nan")
                vals.append(float(value) * scale)
            return vals

        def _has_values(vals):
            return any(math.isfinite(v) for v in vals)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        axes = axes.ravel()

        for split in splits:
            for probe_name in ("Y", "F"):
                key = f"{split}_loss{probe_name}"
                vals = _values(key)
                if _has_values(vals):
                    axes[0].plot(
                        epochs,
                        vals,
                        linestyle=metric_styles[probe_name],
                        color=split_colors[split],
                        label=f"{split} {probe_name}",
                    )
        axes[0].set_title("Loss")
        axes[0].set_ylabel("bits")

        for split in splits:
            for probe_name in ("Y", "F"):
                key = f"{split}_acc{probe_name}"
                vals = _values(key, scale=100.0)
                if _has_values(vals):
                    axes[1].plot(
                        epochs,
                        vals,
                        linestyle=metric_styles[probe_name],
                        color=split_colors[split],
                        label=f"{split} {probe_name}",
                    )
        axes[1].set_title("Accuracy")
        axes[1].set_ylabel("percent")

        for split in splits:
            key = f"{split}_binding_info"
            vals = _values(key)
            if _has_values(vals):
                axes[2].plot(epochs, vals, color=split_colors[split], label=split)
        axes[2].set_title("Binding Info")
        axes[2].set_ylabel("bits")

        for split in splits:
            key = f"{split}_binding_ratio"
            vals = _values(key)
            if _has_values(vals):
                axes[3].plot(epochs, vals, color=split_colors[split], label=split)
        axes[3].set_title("Binding Ratio")
        axes[3].set_ylabel("ratio")

        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)

        title = (
            f"{self.cfg.dataset.name} | {self.cfg.model.short_name} {self.cfg.model.output_mode} | "
            f"Y:{self.cfg.probes.probe_Y.type if self.train_probe_Y else 'off'} "
            f"F:{self.cfg.probes.probe_F.type if self.train_probe_F else 'off'}"
        )
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved training curves to {output_path}")
        return output_path

    def save_model(self, path, update_probeY=True, update_probeF=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            payload = torch.load(path, map_location="cpu")
        else:
            payload = {}
            update_probeY = True
            update_probeF = True
        
        # selectively update probes
        if update_probeY and self.train_probe_Y:
            payload["probe_Y"] = self.probe_Y.state_dict()

        if update_probeF and self.train_probe_F:
            payload["probe_F"] = self.probe_F.state_dict()

        payload["best_val"] = self.best_val
        payload["best_test"] = self.best_test

        torch.save(payload, path)


    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            self._load_checkpoint_dict(ckpt)
        return self

    def _load_checkpoint_dict(self, ckpt):
        """Load probes and metadata from checkpoint dict."""
        if self.train_probe_Y:
            if "probe_Y" not in ckpt:
                raise KeyError("Checkpoint does not contain 'probe_Y' but trainer.train_probe_Y=true.")
            self.probe_Y.load_state_dict(ckpt["probe_Y"])
        if self.train_probe_F:
            if "probe_F" not in ckpt:
                raise KeyError("Checkpoint does not contain 'probe_F' but trainer.train_probe_F=true.")
            self.probe_F.load_state_dict(ckpt["probe_F"])
        self.best_val = ckpt["best_val"]
        self.best_test = ckpt["best_test"]
        if isinstance(self.best_val, list):
            self.best_val = {"Y": self.best_val[0], "F": self.best_val[1]}
        if isinstance(self.best_test, list):
            self.best_test = {"Y": self.best_test[0], "F": self.best_test[1]}
        if not self.train_probe_Y:
            self.best_val["Y"] = None
            self.best_test["Y"] = None
        if not self.train_probe_F:
            self.best_val["F"] = None
            self.best_test["F"] = None
        logger.info(f"Loaded checkpoint (best_val={self.best_val}, best_test={self.best_test})")

    def _maybe_save_probe_weights_if_missing(self):
        """
        Optionally export probe weights to a standalone file, only if absent.
        Controlled by:
          - trainer.save_probe_weights_if_missing (bool, default: False)
          - trainer.probe_weights_path (str, default: <output_dir>/probe_weights.pth)
        """
        enabled = bool(getattr(self.cfg.trainer, "save_probe_weights_if_missing", False))
        if not enabled:
            return

        probe_weights_path = getattr(self.cfg.trainer, "probe_weights_path", None)
        if probe_weights_path is None:
            probe_weights_path = os.path.join(self.output_dir, "probe_weights.pth")
        probe_weights_path = str(probe_weights_path)

        if os.path.exists(probe_weights_path):
            logger.info(f"Probe weights already exist at {probe_weights_path}; skipping export.")
            return

        parent_dir = os.path.dirname(probe_weights_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        payload = {"best_val": self.best_val, "best_test": self.best_test}
        if self.train_probe_Y:
            payload["probe_Y"] = self.probe_Y.state_dict()
        if self.train_probe_F:
            payload["probe_F"] = self.probe_F.state_dict()
        torch.save(payload, probe_weights_path)
        logger.info(f"Saved probe weights to {probe_weights_path}")

    # ----------------------
    # main training loop (train_loop)
    # ----------------------
    def train_loop(self, inputs, Y, F, inputs_are_z=False):
        inputs = inputs.to(self.device, non_blocking=self.pin_memory)
        Y = Y.to(self.device)
        F = F.to(self.device)

        if inputs_are_z:
            Z = inputs
        else:
            with torch.no_grad():
                Z = self.model(inputs)  # [B, z_dim]


        out = {}
        if self.train_probe_Y:
            # predict Y, condition on (Z,Y)
            logits_Y = self.probe_Y(Z, Y)   # [B, nY]
            loss_Y = self.criterion_Y(logits_Y, Y.float())
            out["lossY"] = loss_Y
            with torch.no_grad():
                predY = (logits_Y > 0).float()
                out["accY"] = (predY == Y).float().mean().item()

        if self.train_probe_F:
            # predict F, condition on (Z,F)
            logits_F = self.probe_F(Z, F)   # [B, nF]
            target_F = F.float()
            loss_F = self.criterion_F(logits_F, target_F)
            out["lossF"] = loss_F
            with torch.no_grad():
                predF = (logits_F > 0).float()
                out["accF"] = (predF == target_F).float().mean().item()

        return out



def set_random_seed(seed_value):

    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(cfg):
    
    if cfg.dataset.name == 'colorshape':
        from dataset.ColorShape import ColorShapeDataset
        return ColorShapeDataset
    elif cfg.dataset.name == 'balancedcolorshape':
        from dataset.BalancedColorShape import BalancedColorShapeDataset
        return BalancedColorShapeDataset
    elif cfg.dataset.name == 'clevr':
        from dataset.CLEVR import CLEVRDataset
        return CLEVRDataset
    elif cfg.dataset.name == 'vgcolor':
        from dataset.VGColor import VGColorDataset
        return VGColorDataset
    elif cfg.dataset.name == 'vgtopattr':
        from dataset.VGTopAttr import VGTopAttrDataset
        return VGTopAttrDataset
    elif cfg.dataset.name == 'occlusionclevr':
        from dataset.OcclusionClevr import OcclusionClevrDataset
        return OcclusionClevrDataset
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_shapes(model, model_name):
    logger.info(f"{model_name} parameter shapes:")
    for name, p in model.named_parameters():
        logger.info(f"  {name}: {tuple(p.shape)} ({p.numel():,})")

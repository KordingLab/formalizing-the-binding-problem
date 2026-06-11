
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def load_backbone(cfg):
    if cfg.model.load_from == "timm":
        backbone = timm.create_model(
            cfg.model.name,
            pretrained=getattr(cfg.model, "pretrained", True),
        )
        if hasattr(backbone, "embed_dim"):
            embed_dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            embed_dim = backbone.num_features
        else:
            raise RuntimeError(f"Cannot infer embedding dimension for model {cfg.model.name}")

        if hasattr(backbone, "patch_embed"):
            patch_size = backbone.patch_embed.patch_size
            patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        return backbone, embed_dim, patch_size, None

    if cfg.model.load_from == "huggingface":
        from transformers import AutoModel, AutoImageProcessor

        model = AutoModel.from_pretrained(cfg.model.name)
        image_processor = AutoImageProcessor.from_pretrained(cfg.model.name)

        name = cfg.model.name.lower()
        if "clip" in name and hasattr(model.config, "vision_config"):
            embed_dim = getattr(model.config.vision_config, "hidden_size", None)
            patch_size = getattr(model.config.vision_config, "patch_size", None)
        else:
            embed_dim = getattr(model.config, "hidden_size", None)
            patch_size = getattr(model.config, "patch_size", None)

        if embed_dim is None or patch_size is None:
            raise RuntimeError(f"Cannot infer embedding dimension/patch size for model {cfg.model.name}")

        return model, embed_dim, patch_size, image_processor

    raise ValueError(f"Unknown model.load_from: {cfg.model.load_from}")

class PreprocessorWrapper:
    """
    Normalizes inputs for timm vs huggingface backbones.

    - timm: returns model-normalized image tensor.
    - huggingface: returns dict with pixel_values (+ dummy text for CLIP).
    """
    def __init__(self, cfg, image_processor=None, backbone=None):
        self.cfg = cfg
        self.image_processor = image_processor
        self._is_clip = (cfg.model.load_from == "huggingface") and ("clip" in cfg.model.name.lower())
        self._is_clip_336 = self._is_clip and ("336" in cfg.model.name.lower())
        self._backbone = backbone

        # Normalization stats for timm/HF models. Keep as simple tuples so we can
        # move/cast per batch device/dtype at call time.
        self._timm_mean = None
        self._timm_std = None
        self._hf_mean = None
        self._hf_std = None
        self._hf_do_normalize = True

        if cfg.model.load_from == "timm" and backbone is not None:
            default_cfg = getattr(backbone, "default_cfg", {}) or {}
            self._timm_mean = self._coerce_stats(default_cfg.get("mean", None))
            self._timm_std = self._coerce_stats(default_cfg.get("std", None))
        elif cfg.model.load_from == "huggingface" and image_processor is not None:
            self._hf_mean = self._coerce_stats(getattr(image_processor, "image_mean", None))
            self._hf_std = self._coerce_stats(getattr(image_processor, "image_std", None))
            self._hf_do_normalize = bool(getattr(image_processor, "do_normalize", True))

    @staticmethod
    def _coerce_stats(stats):
        if stats is None:
            return None
        if isinstance(stats, (int, float)):
            return (float(stats),)
        if isinstance(stats, (list, tuple)):
            if len(stats) == 0:
                return None
            return tuple(float(x) for x in stats)
        return None

    @staticmethod
    def _normalize(images: torch.Tensor, mean, std) -> torch.Tensor:
        if mean is None or std is None:
            return images
        if len(mean) != len(std):
            return images
        mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        if mean_t.shape[1] != images.shape[1]:
            # Best-effort fallback for grayscale-style configs.
            if mean_t.shape[1] == 1:
                mean_t = mean_t.expand(1, images.shape[1], 1, 1)
                std_t = std_t.expand(1, images.shape[1], 1, 1)
            else:
                return images
        return (images - mean_t) / std_t

    def __call__(self, images, device=None):
        if device is None:
            device = images.device

        if self.cfg.model.load_from == "timm":
            return self._normalize(images, self._timm_mean, self._timm_std)  # [B,3,H,W]


        # huggingface expects dict inputs
        pixel_values = images

        # CLIP via HF often expects text inputs in the forward; add empty prompt tokens
        if self._is_clip:
            
            if self._is_clip_336:
                # images: (B, 3, 224, 224)
                images_336 = F.interpolate(
                    pixel_values,
                    size=(336, 336),
                    mode="bilinear",
                    align_corners=False
                )
                pixel_values = images_336

        if self._hf_do_normalize:
            pixel_values = self._normalize(pixel_values, self._hf_mean, self._hf_std)

        # images are already tensors in [0,1] -> provide normalized pixel values
        inputs = {"pixel_values": pixel_values}

        if self._is_clip:

            B = images.shape[0]
            input_ids = torch.empty(B, 2, dtype=torch.long, device=device)
            input_ids[:, 0] = 49406  # <|startoftext|>
            input_ids[:, 1] = 49407  # <|endoftext|>
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = torch.ones(B, 2, dtype=torch.long, device=device)

        return inputs
        
class ModelWrapper(nn.Module):
    """
    Frozen vision backbone selected by cfg.model.name.

    output_mode:
      - "cls"              -> [B, D]
      - "mean_spatial"     -> [B, D] = mean(spatial_tokens)
      - "all"              -> [B, N*D]
      - "cls_mean"         -> [B, 2D] = concat(cls, mean(spatial_tokens))
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_mode = cfg.model.output_mode

        self.backbone, self.token_dim, self.patch_size, self.image_processor = load_backbone(cfg)
        self.preproc = PreprocessorWrapper(cfg, self.image_processor, self.backbone)

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.output_mode in {"cls", "mean_spatial"}:
            self.z_dim = self.token_dim
        elif self.output_mode == "all":
            num_patches = (cfg.dataset.image_size // self.patch_size) ** 2
            self.z_dim = (num_patches + 1) * self.token_dim
        elif self.output_mode == "cls_mean":
            self.z_dim = self.token_dim * 2
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def _forward_tokens(self, images):
        """
        images: [B, 3, H, W] float tensor in [0, 1]
        """
        if self.cfg.model.load_from == "huggingface":
            inputs = self.preproc(images, device=images.device)
            out = self.backbone(**inputs)

            # try common places HF models store vision tokens
            tokens = None
            if hasattr(out, "vision_model_output") and out.vision_model_output is not None:
                tokens = getattr(out.vision_model_output, "last_hidden_state", None)
            if tokens is None:
                tokens = getattr(out, "last_hidden_state", None)

            if tokens is None:
                raise RuntimeError("Cannot find vision tokens in HF model output.")
        else:
            # timm
            timm_images = self.preproc(images, device=images.device)
            tokens = self.backbone.forward_features(timm_images)  # [B, N, D] for ViTs

        return tokens

    def format_tokens(self, tokens, output_mode=None):
        if output_mode is None:
            output_mode = self.output_mode

        if output_mode == "all":
            return tokens.reshape(tokens.shape[0], -1)

        cls = tokens[:, 0]  # [B, D]
        if output_mode == "cls":
            return cls

        if tokens.shape[1] > 1:
            spatial = tokens[:, 1:]
            mean_spatial = spatial.mean(dim=1)
        else:
            mean_spatial = cls

        if output_mode == "mean_spatial":
            return mean_spatial
        if output_mode == "cls_mean":
            return torch.cat([cls, mean_spatial], dim=-1)

        raise ValueError(f"Unknown output_mode: {output_mode}")

    def format_flattened_tokens(self, flat_tokens, output_mode=None):
        if flat_tokens.ndim != 2:
            raise ValueError(f"Expected flattened tokens with shape [B, N*D], got {tuple(flat_tokens.shape)}.")
        if flat_tokens.shape[1] % int(self.token_dim) != 0:
            raise ValueError(
                f"Flattened token dim {flat_tokens.shape[1]} is not divisible by token_dim={int(self.token_dim)}."
            )
        tokens = flat_tokens.view(flat_tokens.shape[0], -1, int(self.token_dim))
        return self.format_tokens(tokens, output_mode=output_mode)

    @torch.no_grad()
    def forward_all_tokens(self, images):
        tokens = self._forward_tokens(images)
        return tokens.reshape(tokens.shape[0], -1)

    @torch.no_grad()
    def forward(self, images):
        tokens = self._forward_tokens(images)
        return self.format_tokens(tokens)

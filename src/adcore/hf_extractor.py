"""A frozen Hugging Face ViT's hidden states as patch tokens, with a feature cache.

Needs the ``hf`` extra (``transformers``).
"""

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor, nn


class HFViTExtractor(nn.Module):
    """Patch tokens ``(B, h*w, C)`` of a frozen Hugging Face ViT (e.g. DINOv2, DINOv3).

    Args:
        model_tag: Hub name, e.g. ``"facebook/dinov2-with-registers-giant"``.
        hidden_states: Hidden states to read: 0 is the patch embedding, i the output of
            block i. The blocks after the last one are dropped.
        combine: ``"mean"`` averages the states (C = width); ``"concat"`` stacks them
            along the channels in the order listed (C = len(hidden_states) * width).
        cache_gb: In-memory cache of test-time features, by image content, oldest
            dropped first beyond this size. 0 disables it.
        cache_dir: On-disk cache, one file per image, under a subdirectory named by the
            settings that change the features; support (training) features go to its
            ``support/`` subdirectory. Defaults to ``$ADCORE_FEATURE_CACHE``, so the
            cache stays out of configs; None and unset disable it.

    CLS and register tokens are dropped. ``num_states`` and ``state_width`` describe the
    channel layout for ``combine="concat"``.
    """

    def __init__(
        self,
        model_tag: str,
        hidden_states: list[int],
        combine: str = "mean",
        cache_gb: float = 0.0,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        if combine not in ("mean", "concat"):
            raise ValueError(f"combine must be 'mean' or 'concat', got {combine!r}")
        self.model_tag = model_tag
        self.hidden_states = list(hidden_states)
        self.combine = combine
        self.backbone = AutoModel.from_pretrained(model_tag)
        self.backbone.requires_grad_(False)
        # DINOv2 keeps its blocks in .encoder, DINOv3 in .model
        encoder = self.backbone.encoder if hasattr(self.backbone, "encoder") else self.backbone.model
        encoder.layer = encoder.layer[: max(self.hidden_states)]
        config = self.backbone.config
        self.patch_size = config.patch_size
        self.num_prefix_tokens = 1 + getattr(config, "num_register_tokens", 0)
        self.num_states = len(self.hidden_states) if combine == "concat" else 1
        self.state_width = config.hidden_size

        self.cache_bytes = int(cache_gb * 2**30)
        self._cache: OrderedDict[bytes, Tensor] = OrderedDict()
        self._cached_bytes = 0
        cache_dir = cache_dir or os.environ.get("ADCORE_FEATURE_CACHE")
        self.cache_dir = None
        if cache_dir:
            text = json.dumps(self.cache_settings(), sort_keys=True)
            self.cache_dir = Path(cache_dir) / hashlib.sha256(text.encode()).hexdigest()[:12]
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "settings.json").write_text(text)

    def cache_settings(self) -> dict:
        """What picks the cache subdirectory. The layout is that of semdino's former
        SubSpaceADExtractor, so its caches stay valid: a contiguous mean is keyed as the
        range ``layers``, anything else by its ``hidden_states`` (and ``combine``)."""
        states = self.hidden_states
        contiguous = states == list(range(states[0], states[-1] + 1))
        settings = {
            "model_tag": self.model_tag,
            "layers": [states[0], states[-1] + 1] if contiguous and self.combine == "mean" else [23, 30],
            "grid": False,
            "tile_size": None,
            "tile_overlap": None,
            "dtype": None,
        }
        if not (contiguous and self.combine == "mean"):
            settings["hidden_states"] = states
        if self.combine != "mean":
            settings["combine"] = self.combine
        return settings

    def train(self, mode: bool = True) -> "HFViTExtractor":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def embed(self, images: Tensor) -> Tensor:
        """``(B, h*w, C)`` features, without the cache."""
        outputs = self.backbone(pixel_values=images, output_hidden_states=True)
        features = torch.stack([outputs.hidden_states[i] for i in self.hidden_states], dim=1)
        features = features[:, :, self.num_prefix_tokens :].float()
        if self.combine == "concat":
            return features.permute(0, 2, 1, 3).flatten(2)
        return features.mean(dim=1)

    def forward(self, images: Tensor) -> Tensor:
        if self.cache_dir is None and (self.training or not self.cache_bytes):
            return self.embed(images)
        keys = [hashlib.blake2b(x.cpu().numpy().tobytes()).digest() for x in images]
        found = [self._load(key) for key in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        if missing:
            # Cloned so each cached tensor owns its storage rather than viewing the batch's
            computed = [value.clone() for value in self.embed(images[missing]).cpu()]
            for i, value in zip(missing, computed):
                found[i] = value
        for i, (key, value) in enumerate(zip(keys, found)):
            self._store(key, value, computed=i in missing)
        return torch.stack(found).to(images.device)

    def _path(self, key: bytes) -> Path:
        directory = self.cache_dir / "support" if self.training else self.cache_dir
        return directory / f"{key.hex()}.pt"

    def _load(self, key: bytes) -> Tensor | None:
        if key in self._cache:
            return self._cache[key]
        if self.cache_dir is not None:
            path = self._path(key)
            if path.exists():
                return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def _store(self, key: bytes, value: Tensor, computed: bool) -> None:
        if computed and self.cache_dir is not None:
            # Written then renamed, so a reader in another process never sees half a file
            path = self._path(key)
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(value, tmp)
            os.replace(tmp, path)
        if self.cache_bytes and not self.training and key not in self._cache:
            self._cache[key] = value
            self._cached_bytes += value.nbytes
            while self._cached_bytes > self.cache_bytes:
                self._cached_bytes -= self._cache.popitem(last=False)[1].nbytes

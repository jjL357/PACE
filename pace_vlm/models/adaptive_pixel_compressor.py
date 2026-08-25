"""Adaptive Pixel Compressor (APC) from PACE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class APCResult:
    """Output of adaptive pixel condensation for one image."""

    image: Image.Image
    retention_ratio: float
    global_information_density: float
    local_detail_contrast: float


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 2048 * 28 * 28,
) -> Tuple[int, int]:
    """Match Qwen2.5-VL's bounded, factor-aligned resize policy."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "The absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width):.2f}."
        )

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def resize_to_pixel_bounds(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """Resize an image to the fixed or dynamic pixel range used for evaluation."""
    height, width = smart_resize(
        image.height,
        image.width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (height, width) == (image.height, image.width):
        return image
    return image.resize((width, height), Image.Resampling.BICUBIC)


def compressed_resolution(
    height: int,
    width: int,
    patch_size: int,
    retention_ratio: float,
) -> Tuple[int, int, float]:
    """Find a patch-aligned resolution closest to the target token ratio."""
    original_patch_height = math.ceil(height / patch_size)
    original_patch_width = math.ceil(width / patch_size)
    original_tokens = original_patch_height * original_patch_width
    target_tokens = max(1, round(original_tokens * retention_ratio))

    aspect_ratio = height / width
    ideal_patch_width = math.sqrt(target_tokens / aspect_ratio)
    ideal_patch_height = ideal_patch_width * aspect_ratio
    candidates = (
        (max(1, math.floor(ideal_patch_height)), max(1, math.floor(ideal_patch_width))),
        (max(1, math.floor(ideal_patch_height)), max(1, math.ceil(ideal_patch_width))),
        (max(1, math.ceil(ideal_patch_height)), max(1, math.floor(ideal_patch_width))),
        (max(1, math.ceil(ideal_patch_height)), max(1, math.ceil(ideal_patch_width))),
    )
    patch_height, patch_width = min(
        candidates,
        key=lambda item: (
            abs(item[0] * item[1] - target_tokens),
            abs(item[0] / item[1] - aspect_ratio),
        ),
    )
    actual_ratio = patch_height * patch_width / original_tokens
    return patch_height * patch_size, patch_width * patch_size, actual_ratio


class ShallowFeaturePreview(nn.Module):
    """Run the first K ViT blocks without executing the full vision encoder."""

    def __init__(self, vision_model: nn.Module, depth: int = 1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Preview depth must be at least 1, got {depth}.")
        if depth > len(vision_model.blocks):
            raise ValueError(
                f"Preview depth {depth} exceeds {len(vision_model.blocks)} ViT blocks."
            )
        self.vision_model = vision_model
        self.depth = depth

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        vision_model = self.vision_model
        hidden_states = vision_model.patch_embed(pixel_values)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        grid_thw = grid_thw.to(device=hidden_states.device, dtype=torch.int64)

        rotary_pos_emb = vision_model.rot_pos_emb(grid_thw)
        window_index, window_cu_seqlens = vision_model.get_window_index(grid_thw)
        sequence_length = hidden_states.shape[0]

        hidden_states = hidden_states.reshape(
            sequence_length // vision_model.spatial_merge_unit,
            vision_model.spatial_merge_unit,
            -1,
        )[window_index].reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            sequence_length // vision_model.spatial_merge_unit,
            vision_model.spatial_merge_unit,
            -1,
        )[window_index].reshape(sequence_length, -1)
        embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())

        window_cu_seqlens = torch.as_tensor(
            window_cu_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        ).unique_consecutive()
        full_cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(0, dtype=torch.int32)
        full_cu_seqlens = F.pad(full_cu_seqlens, (1, 0), value=0)

        for block_index in range(self.depth):
            cu_seqlens = (
                full_cu_seqlens
                if block_index in vision_model.fullatt_block_indexes
                else window_cu_seqlens
            )
            block_output = vision_model.blocks[block_index](
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            hidden_states = (
                block_output[0] if isinstance(block_output, (tuple, list)) else block_output
            )
        return hidden_states


class AdaptivePixelCompressor(nn.Module):
    """Estimate information density and continuously resize the input image."""

    def __init__(
        self,
        vision_model: nn.Module,
        preview_depth: int = 1,
        global_weight: float = 0.6,
        detail_fraction: float = 0.1,
        detail_scale: float = 1.5,
        minimum_retention: float = 0.05,
        patch_size: int = 28,
    ) -> None:
        super().__init__()
        if not 0.0 <= global_weight <= 1.0:
            raise ValueError("global_weight must be in [0, 1].")
        if not 0.0 < detail_fraction <= 1.0:
            raise ValueError("detail_fraction must be in (0, 1].")
        if detail_scale <= 0.0:
            raise ValueError("detail_scale must be positive.")
        if not 0.0 < minimum_retention <= 1.0:
            raise ValueError("minimum_retention must be in (0, 1].")

        self.preview = ShallowFeaturePreview(vision_model, preview_depth)
        self.global_weight = global_weight
        self.detail_fraction = detail_fraction
        self.detail_scale = detail_scale
        self.minimum_retention = minimum_retention
        self.patch_size = patch_size

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(features.reshape(features.shape[0], -1), p=2, dim=-1, eps=1e-12)

    def score(self, features: torch.Tensor) -> Tuple[float, float, float]:
        """Return retention ratio, global density, and local detail contrast."""
        features = self._normalize(features)
        token_count = features.shape[0]
        if token_count <= 1:
            return self.minimum_retention, 0.0, 0.0

        feature_sum = features.sum(dim=0)
        self_similarity = features.square().sum()
        pairwise_similarity = (torch.dot(feature_sum, feature_sum) - self_similarity) / (
            token_count * (token_count - 1)
        )
        global_density = (1.0 - pairwise_similarity).clamp(0.0, 1.0)

        background = F.normalize((feature_sum / token_count).unsqueeze(0), p=2, dim=-1, eps=1e-12)
        distances = torch.linalg.vector_norm(features - background, dim=-1)
        detail_count = min(
            token_count,
            max(1, math.ceil(token_count * self.detail_fraction)),
        )
        top_detail_distance = distances.topk(detail_count).values.mean()
        local_detail = (top_detail_distance / self.detail_scale).clamp(0.0, 1.0)

        retention = (
            self.global_weight * global_density + (1.0 - self.global_weight) * local_detail
        ).clamp(self.minimum_retention, 1.0)
        return retention.item(), global_density.item(), local_detail.item()

    @torch.no_grad()
    def compress(
        self,
        image: Image.Image,
        processor,
        device: torch.device,
        maximum_retention: Optional[float] = None,
    ) -> APCResult:
        """Run the shallow preview and resize one image with bicubic interpolation."""
        image = image.convert("RGB")
        prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
        vision_dtype = next(self.preview.vision_model.parameters()).dtype
        features = self.preview(
            inputs["pixel_values"].to(dtype=vision_dtype),
            inputs["image_grid_thw"],
        )
        retention, global_density, local_detail = self.score(features)
        if maximum_retention is not None:
            retention = min(retention, maximum_retention)

        height, width, actual_retention = compressed_resolution(
            image.height,
            image.width,
            self.patch_size,
            retention,
        )
        compressed = image.resize((width, height), Image.Resampling.BICUBIC)
        return APCResult(
            image=compressed,
            retention_ratio=actual_retention,
            global_information_density=global_density,
            local_detail_contrast=local_detail,
        )

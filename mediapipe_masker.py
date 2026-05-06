"""MediaPipe-based mask generation for Reference Latent+.

Adapted from comfyui-wan-i2v-control's masker. Generates per-region masks
from the selfie_multiclass segmentation model (face / hair / body / clothes /
background), with optional grow / feather / ignore-area post-processing.

Models are stored in ``<ComfyUI>/models/mediapipe/`` and auto-downloaded on
first use. MediaPipe is an optional dependency — if not installed,
``generate_auto_mask`` returns ``None`` and the node falls back to unmasked
behaviour.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from scipy.ndimage import distance_transform_edt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# --- Model download ---------------------------------------------------

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
_MODEL_NAME = "selfie_multiclass_256x256.tflite"


def _get_model_path() -> str:
    """Return path to the selfie_multiclass model, downloading if missing."""
    import folder_paths
    models_dir = os.path.join(folder_paths.models_dir, "mediapipe")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, _MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"[RefLatentPlus] Downloading MediaPipe model: {_MODEL_NAME}")
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print(f"[RefLatentPlus] Downloaded to: {model_path}")
    return model_path


# selfie_multiclass class indices.
_SEG_BACKGROUND = 0
_SEG_HAIR = 1
_SEG_BODY = 2
_SEG_FACE = 3
_SEG_CLOTHES = 4


# --- Mask generation --------------------------------------------------

def _person_segmentation_mask(
    image_np: np.ndarray, face: bool, hair: bool, body: bool, clothes: bool, background: bool,
) -> np.ndarray:
    """Run MediaPipe selfie_multiclass and combine selected categories.

    Returns a ``[H, W]`` float32 mask in 0/1.
    """
    model_path = _get_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    h, w = image_np.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        result = segmenter.segment(mp_image)
        if not result.category_mask:
            return out
        cat = result.category_mask.numpy_view()
        if cat.ndim == 3:
            cat = cat[:, :, 0]
        if background:
            out = np.where(cat == _SEG_BACKGROUND, 1.0, out)
        if hair:
            out = np.where(cat == _SEG_HAIR, 1.0, out)
        if body:
            out = np.where(cat == _SEG_BODY, 1.0, out)
        if face:
            out = np.where(cat == _SEG_FACE, 1.0, out)
        if clothes:
            out = np.where(cat == _SEG_CLOTHES, 1.0, out)
    return out


def _apply_ignore_area(mask: torch.Tensor, ignore_area: str, ignore_percent: float) -> torch.Tensor:
    """Zero out (= "exclude from ref") a side strip of the mask."""
    if ignore_area == "none":
        return mask
    _b, h, w = mask.shape
    if ignore_area in ("left", "right"):
        n = int(ignore_percent * w)
        if ignore_area == "left":
            mask[:, :, :n] = 0.0
        else:
            mask[:, :, -n:] = 0.0
    elif ignore_area in ("top", "bottom"):
        n = int(ignore_percent * h)
        if ignore_area == "top":
            mask[:, :n, :] = 0.0
        else:
            mask[:, -n:, :] = 0.0
    return mask


def _apply_grow(mask: torch.Tensor, grow: int) -> torch.Tensor:
    """Grow (>0) or shrink (<0) the mask's white area by N pixels.

    Uses scipy's Euclidean distance transform when available — O(N)
    regardless of grow size, ~40× faster than max-pool at 100+ pixel grows.
    Produces circular (rounded) dilation instead of square (Chebyshev),
    which is the visually nicer behaviour for mask use cases anyway.

    Falls back to separable max-pool if scipy isn't installed.
    """
    grow = int(grow)
    if grow == 0:
        return mask
    pixels = abs(grow)

    if _SCIPY_AVAILABLE:
        return _apply_grow_dt(mask, grow, pixels)
    return _apply_grow_maxpool(mask, grow, pixels)


def _apply_grow_dt(mask: torch.Tensor, grow: int, pixels: int) -> torch.Tensor:
    """Distance-transform path. Circular dilation/erosion at fixed cost in N."""
    out = torch.empty_like(mask)
    for b in range(mask.shape[0]):
        m_np = (mask[b] > 0.5).cpu().numpy()
        if grow > 0:
            # Dilate: every pixel within `pixels` of a white pixel becomes white.
            dt = distance_transform_edt(~m_np)
            out[b] = torch.from_numpy((dt <= pixels).astype(np.float32))
        else:
            # Erode: every pixel within `pixels` of a black pixel becomes black.
            dt = distance_transform_edt(m_np)
            out[b] = torch.from_numpy((dt > pixels).astype(np.float32))
    return out.to(mask.device)


def _apply_grow_maxpool(mask: torch.Tensor, grow: int, pixels: int) -> torch.Tensor:
    """Fallback when scipy isn't installed. Square (Chebyshev) dilation via
    separable max-pool. Slow at large kernels but still correct."""
    kernel = pixels * 2 + 1
    pad = pixels
    m = mask.unsqueeze(1)
    sign = 1 if grow > 0 else -1
    if sign < 0:
        m = -m
    m = F.pad(m, (pad, pad, 0, 0), mode="replicate")
    m = F.max_pool2d(m, (1, kernel), stride=1, padding=0)
    m = F.pad(m, (0, 0, pad, pad), mode="replicate")
    m = F.max_pool2d(m, (kernel, 1), stride=1, padding=0)
    if sign < 0:
        m = -m
    return m.squeeze(1)


def _apply_feather(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Soften mask edges by ``feather`` pixels via separable box blur.

    ``feather`` is a pixel radius — at feather=10, edges blur over roughly 10
    pixels. Implementation uses two separable passes (O(K) per pixel) which
    approximates a Gaussian via central-limit theorem on box filters. Fast
    even at 100+ pixel radii.
    """
    feather = int(feather)
    if feather <= 0:
        return mask
    kernel = feather * 2 + 1  # always odd
    pad = kernel // 2
    m = mask.unsqueeze(1)  # [B, 1, H, W]
    for _ in range(2):
        m = F.pad(m, (pad, pad, 0, 0), mode="replicate")
        m = F.avg_pool2d(m, (1, kernel), stride=1, padding=0)
        m = F.pad(m, (0, 0, pad, pad), mode="replicate")
        m = F.avg_pool2d(m, (kernel, 1), stride=1, padding=0)
    return m.squeeze(1).clamp(0.0, 1.0)


def generate_auto_mask(
    image: torch.Tensor,
    *,
    face: bool = False, hair: bool = False, body: bool = False,
    clothes: bool = False, background: bool = False,
    ignore_area: str = "none",
    ignore_percent: float = 0.5,
    feather: int = 0,
    grow: int = 0,
) -> Optional[torch.Tensor]:
    """Generate a MASK tensor from an IMAGE via MediaPipe person segmentation.

    Returns ``None`` if no regions are selected (all booleans False) or if
    MediaPipe isn't installed. Mask shape: ``[B, H, W]`` matching ``image``.
    """
    if not any([face, hair, body, clothes, background]):
        return None
    if not MEDIAPIPE_AVAILABLE:
        print("[RefLatentPlus] MediaPipe not installed — auto-mask requested but skipped. "
              "Install with: pip install mediapipe")
        return None

    img_np = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    seg = _person_segmentation_mask(img_np, face, hair, body, clothes, background)
    mask = torch.from_numpy(seg).float().unsqueeze(0).to(image.device)
    mask = _apply_ignore_area(mask, ignore_area, ignore_percent)
    mask = _apply_grow(mask, grow)
    mask = _apply_feather(mask, feather)
    return mask.clamp(0.0, 1.0)

"""Reference Latent+ — improved ReferenceLatent for any model that consumes
``reference_latents`` (Flux, Flux2/Klein, Lumina2, Z-Image, Wan, Hunyuan,
Qwen-Image, …).

Replaces the stock ``ReferenceLatent`` node, addressing six concrete
weaknesses: signed per-image strength, per-image timestep gating, per-image
compositional masking (with built-in MediaPipe auto-mask for face / body /
clothes / background), attention cost via megapixel cap, and 1-4 image
inputs (no manual VAE Encode wiring).

Empirically tuned on Flux2/Klein 9B; defaults reflect that. On other models
the same mechanism applies but the sweet-spot values may differ.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import node_helpers

from .mediapipe_masker import generate_auto_mask, MEDIAPIPE_AVAILABLE


# We only ship the "index" (Independent) ref-latents-method.
# Empirically tested on Klein 9B (Base + Distilled): it's the one that behaves
# like users expect — refs as separate guides, not part of the scene. Other
# Flux family models that read this key (Flux1, LongCatImage) accept it; non-
# Flux models that consume `reference_latents` (Wan, Lumina, Hunyuan, Qwen-
# Image) ignore the method key entirely.
_REF_METHOD = "index"


# Per-image auto-mask region keys — match the kwargs of generate_auto_mask().
# Person segmentation only (face / hair / body / clothes / background).
# Sub-face landmarks (eyes / lips / pupils / etc.) were removed as overkill.
_REGION_KEYS = ("face", "hair", "body", "clothes", "background")


def _cap_megapixels(image: torch.Tensor, max_mp: float) -> torch.Tensor:
    """Downscale image to fit ``max_mp`` megapixels, preserving aspect ratio.

    Never upscales — small images pass through unchanged. Final dimensions
    are rounded down to multiples of 16 (covers Flux/Klein VAE's 8× downsample
    + 2× patch, plus most other VAEs which are 8× — divisible by 16 is safe
    universally).

    Input/output: ComfyUI IMAGE tensor ``[B, H, W, C]`` in ``[0, 1]``.
    """
    _b, h, w, _c = image.shape
    current_mp = (h * w) / 1_000_000.0
    if current_mp <= max_mp:
        return image
    scale = (max_mp / current_mp) ** 0.5
    new_h = max(16, (int(h * scale) // 16) * 16)
    new_w = max(16, (int(w * scale) // 16) * 16)
    img_chw = image.permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(img_chw, size=(new_h, new_w), mode="bicubic", align_corners=False)
    return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0).contiguous()


def _resize_mask_to_image(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Normalise a MASK to ``[B, H, W]`` matching the image's spatial size and batch."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1)
    _b, h, w, _c = image.shape
    if mask.shape[-2:] != (h, w):
        mask = F.interpolate(
            mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze(1)
    return mask.clamp(0.0, 1.0)


def _apply_mask_pixel_grey(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked-out (mask=0) pixels with mid-grey (0.5)."""
    mask = _resize_mask_to_image(mask, image).unsqueeze(-1)  # [B, H, W, 1]
    return image * mask + 0.5 * (1.0 - mask)


def _resize_mask_to_latent(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Return mask resized to latent's spatial dims, broadcastable as [B, 1, lH, lW]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    _b, _c, lh, lw = latent.shape
    if mask.shape[-2:] != (lh, lw):
        mask = F.interpolate(
            mask.unsqueeze(1).float(), size=(lh, lw), mode="bilinear", align_corners=False,
        ).squeeze(1)
    return mask.clamp(0.0, 1.0).to(latent.dtype).to(latent.device).unsqueeze(1)


def _apply_mask_latent_zero(latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out latent positions where the (resized) mask is 0.

    Latent shape: ``[B, C, H/16, W/16]``. Multiplied positions become near-zero
    (modulo VAE-induced offset). Masked tokens have constant K/V = bias terms,
    so attention distributes uniformly over them — clean "ignore" semantics.
    """
    return latent * _resize_mask_to_latent(mask, latent)


def _apply_mask_latent_noise(latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace latent positions where mask=0 with random Gaussian noise.

    Different statistical shape from ``latent_zero``: instead of constant
    masked tokens, each masked position has independent random K/V. Model
    attends weakly to noise because Q rarely matches a random K. The "noise"
    pattern is fixed (seed=0) so the same node call produces the same noise
    for reproducibility within a workflow.
    """
    m1c = _resize_mask_to_latent(mask, latent)
    gen = torch.Generator(device=latent.device).manual_seed(0)
    noise = torch.randn(latent.shape, dtype=latent.dtype, device=latent.device, generator=gen)
    return latent * m1c + noise * (1.0 - m1c)


_MASK_FILL_MODES = ("pixel_grey", "latent_zero", "latent_noise")


def _per_image_inputs(prefix: str) -> dict:
    """Build the per-image config block (`image1`, `image2`, ...).

    Per-image strength + 5 region booleans + 3 mask tuning controls + 2 timestep
    controls. Naming uses the prefix so all of image N's controls group together
    in ComfyUI's UI (order = dict-insertion order in INPUT_TYPES).
    """
    region_tooltips = {
        "face": "Face skin",
        "hair": "Hair",
        "body": "Body skin",
        "clothes": "Clothes",
        "background": "Background (tick to use the background only — handy for environment refs)",
    }
    out: dict = {}
    out[f"{prefix}_strength"] = ("FLOAT", {
        "default": 0.85, "min": -5.0, "max": 50.0, "step": 0.05,
        "tooltip": "Per-image strength (signed). 1.0 = stock ReferenceLatent strength. "
                   "0.85 = empirical sweet spot. 0 = bypass this image entirely. "
                   "Negative = anti-reference (model pulled away from this ref's features).",
    })
    for region in _REGION_KEYS:
        out[f"{prefix}_{region}"] = ("BOOLEAN", {
            "default": False,
            "tooltip": region_tooltips[region],
        })
    out[f"{prefix}_ignore_area"] = (["none", "left", "right", "top", "bottom"], {
        "default": "none",
        "tooltip": "Exclude a side strip from the auto-mask. Useful when multiple subjects are in frame.",
    })
    out[f"{prefix}_feather"] = ("INT", {
        "default": 0, "min": 0, "max": 200, "step": 1,
        "tooltip": "Soften mask edges by N pixels (0 = sharp). 10-30 is usually enough.",
    })
    out[f"{prefix}_grow"] = ("INT", {
        "default": 0, "min": -200, "max": 200, "step": 1,
        "tooltip": "Grow (>0) or shrink (<0) the mask by N pixels. Useful for tucking "
                   "or extending the masked area beyond what MediaPipe detected.",
    })
    out[f"{prefix}_start_percent"] = ("FLOAT", {
        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
        "tooltip": "Fraction of denoising at which this image's ref starts applying. "
                   "0 = active from the very first step.",
    })
    out[f"{prefix}_end_percent"] = ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
        "tooltip": "Fraction of denoising at which this image's ref stops applying. "
                   "1 = active through the last step. For Flux2/Klein 9B, style/"
                   "identity signal concentrates at the clean end of denoising — try "
                   "start=0.6, end=1.0 to apply the ref only at the late (clean) "
                   "timesteps and let the noisy/early timesteps freely follow your "
                   "prompt's composition.",
    })
    return out


def _gather_mask_kwargs(prefix: str, kwargs: dict) -> dict:
    """Pull this image's auto-mask config out of the apply() **kwargs dump."""
    mk = {}
    for region in _REGION_KEYS:
        mk[region] = bool(kwargs.get(f"{prefix}_{region}", False))
    mk["ignore_area"] = str(kwargs.get(f"{prefix}_ignore_area", "none"))
    mk["feather"] = int(kwargs.get(f"{prefix}_feather", 0))
    mk["grow"] = int(kwargs.get(f"{prefix}_grow", 0))
    return mk


def _gather_timestep_range(prefix: str, kwargs: dict) -> tuple[float, float]:
    """Pull this image's per-image timestep range from kwargs. Defaults (0, 1)."""
    return (
        float(kwargs.get(f"{prefix}_start_percent", 0.0)),
        float(kwargs.get(f"{prefix}_end_percent", 1.0)),
    )


def _gather_strength(prefix: str, kwargs: dict) -> float:
    """Pull this image's strength scalar from kwargs. Default 0.85."""
    return float(kwargs.get(f"{prefix}_strength", 0.85))


class ReferenceLatentPlus:
    """Reference Latent+ — drop-in replacement for ReferenceLatent."""

    @classmethod
    def INPUT_TYPES(cls):
        optional: dict = {
            "image_2": ("IMAGE",),
            "image_3": ("IMAGE",),
            "image_4": ("IMAGE",),
            "mask_1": ("MASK", {
                "tooltip": "Optional explicit mask for image 1. Overrides this image's "
                           "auto-mask config below.",
            }),
            "mask_2": ("MASK", {"tooltip": "Optional explicit mask for image 2 (overrides auto-mask)."}),
            "mask_3": ("MASK", {"tooltip": "Optional explicit mask for image 3 (overrides auto-mask)."}),
            "mask_4": ("MASK", {"tooltip": "Optional explicit mask for image 4 (overrides auto-mask)."}),
        }
        # Per-image auto-mask config (image1..image4). MediaPipe-driven; ticked
        # regions produce a mask for that image. Explicit mask_N input takes
        # precedence over this config.
        for i in range(1, 5):
            optional.update(_per_image_inputs(f"image{i}"))

        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image_1": ("IMAGE",),
                "max_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                    "tooltip": "Caps each ref image by area. Downscales only — small images "
                               "pass through. Lower = faster (fewer attention tokens).",
                }),
                "mask_fill_mode": (list(_MASK_FILL_MODES), {
                    "default": "pixel_grey",
                    "tooltip": "How masked-out regions are neutralised before the model attends "
                               "to the ref.\n"
                               "  pixel_grey: replace masked pixels with 0.5 grey before VAE "
                               "encode. In-distribution; the masked latent has soft uniform "
                               "signal.\n"
                               "  latent_zero: encode full image, then zero the latent at "
                               "masked positions. Constant K/V at those tokens (= layer bias) "
                               "so attention distributes uniformly over them. Spiritual "
                               "equivalent of 'transparent'.\n"
                               "  latent_noise: same as latent_zero but fills with Gaussian "
                               "noise instead of zeros. Different attention shape — random "
                               "K/V means no Q matches strongly, model weakly ignores those "
                               "tokens.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "advanced/conditioning"

    def apply(
        self,
        conditioning,
        vae,
        image_1,
        max_megapixels: float,
        mask_fill_mode: str = "pixel_grey",
        image_2=None,
        image_3=None,
        image_4=None,
        mask_1=None,
        mask_2=None,
        mask_3=None,
        mask_4=None,
        **kwargs,  # image1_*..image4_* per-image strength, mask, timestep config
    ):
        # Encode each image and collect (latent, start_percent, end_percent).
        # Each image's timestep range and strength are per-image.
        encoded_refs: list[tuple[torch.Tensor, float, float]] = []
        per_image = (
            (1, image_1, mask_1),
            (2, image_2, mask_2),
            (3, image_3, mask_3),
            (4, image_4, mask_4),
        )

        for idx, img, explicit_mask in per_image:
            if img is None:
                continue

            # Per-image strength. 0 = bypass this image entirely.
            img_strength = _gather_strength(f"image{idx}", kwargs)
            if img_strength == 0.0:
                continue

            # Per-image timestep range. Default (0, 1) = always active.
            eff_start, eff_end = _gather_timestep_range(f"image{idx}", kwargs)
            if eff_start >= eff_end:
                # Empty range — this image's ref would never be active. Skip.
                continue

            # Resolve mask precedence: explicit mask_N > auto-mask from regions > none.
            mask = explicit_mask
            if mask is None:
                mask_kwargs = _gather_mask_kwargs(f"image{idx}", kwargs)
                mask = generate_auto_mask(img, **mask_kwargs)

            # Pixel-space modes apply before VAE encode; latent-space modes after.
            if mask is not None and mask_fill_mode == "pixel_grey":
                img = _apply_mask_pixel_grey(img, mask)

            img = _cap_megapixels(img, max_megapixels)

            latent = vae.encode(img[:, :, :, :3])

            if mask is not None and mask_fill_mode == "latent_zero":
                latent = _apply_mask_latent_zero(latent, mask)
            elif mask is not None and mask_fill_mode == "latent_noise":
                latent = _apply_mask_latent_noise(latent, mask)

            if img_strength != 1.0:
                latent = latent * img_strength
            encoded_refs.append((latent, eff_start, eff_end))

        if not encoded_refs:
            return (conditioning,)

        # Partition the timestep schedule by all ref boundaries so each region
        # has exactly one set of active refs (no double-counting via averaging
        # of overlapping conditioning entries).
        breakpoints = sorted({0.0, 1.0} |
                              {s for _, s, _ in encoded_refs} |
                              {e for _, _, e in encoded_refs})

        out: list = []
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            if a >= b:
                continue
            mid = 0.5 * (a + b)
            active = [lat for lat, s, e in encoded_refs if s <= mid < e]

            # Build a conditioning entry for this region.
            region = node_helpers.conditioning_set_values(
                conditioning,
                {"start_percent": float(a), "end_percent": float(b)},
            )
            if active:
                region = node_helpers.conditioning_set_values(
                    region, {"reference_latents": active}, append=True,
                )
                # Conditioning-dict key is `reference_latents_method` (full word).
                region = node_helpers.conditioning_set_values(
                    region, {"reference_latents_method": _REF_METHOD},
                )
            out.extend(region)

        # If the breakpoints exactly cover [0, 1] (always true since we seed with 0
        # and 1) we return as-is. Otherwise we'd need a prompt-only fallback for
        # gaps, but that path can't be reached.
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ReferenceLatentPlus": ReferenceLatentPlus,
    # Backward-compat alias for workflows saved under the old class name.
    # Maps the old key to the new class so existing workflows keep loading.
    "KleinReferenceLatentPlus": ReferenceLatentPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceLatentPlus": "Reference Latent+",
    "KleinReferenceLatentPlus": "Reference Latent+",
}

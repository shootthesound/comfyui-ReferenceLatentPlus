"""comfyui-ReferenceLatentPlus — improved ReferenceLatent for any model that
consumes ``reference_latents`` (Flux, Flux2/Klein, Lumina2, Z-Image, Wan,
Hunyuan, Qwen-Image, …). Empirically tuned on Flux2/Klein 9B."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Tells ComfyUI to serve files in ./web/ (collapsible-groups JS extension).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

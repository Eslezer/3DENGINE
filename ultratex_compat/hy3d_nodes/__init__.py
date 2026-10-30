"""ComfyUI nodes used by ultratex_engine (installed into ComfyUI/custom_nodes/hy3d_nodes)."""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


class Hy3dPinFront:
    """Init views + keep mask that pin UltraTex's front view to the reference image.

    UltraTex conditions on the reference but repaints it, so logos, eyes and colours drift.
    Pinning the front (RePaint keep mask on every step) makes the sampler paint the other five
    views around the real front instead of its own reinterpretation of it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": ("ULTRATEX_PREP",),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
                "erode_px": ("INT", {"default": 6, "min": 0, "max": 256, "tooltip": "Unpinned band inside the outline (at the sampling resolution), where alignment is least reliable."}),
                "feather_px": ("INT", {"default": 12, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("init_views", "keep_mask")
    FUNCTION = "pin"
    CATEGORY = "hy3d"

    def pin(self, prep, resolution, erode_px, feather_px):
        init, keep = pin_front(prep.reference_rgba, prep.normal_rgba[0, ..., 3], resolution, erode_px, feather_px)
        return (torch.from_numpy(init.astype(np.float32) / 255.0), torch.from_numpy(keep))


def pin_front(reference_rgba, front_coverage, resolution, erode_px=6, feather_px=12):
    """(6, R, R, 3) uint8 init views (white except the front) and (6, R, R) float keep masks."""
    r = resolution
    ref = np.asarray(Image.fromarray(reference_rgba).resize((r, r), Image.Resampling.LANCZOS)).astype(np.float32)
    sil = np.asarray(Image.fromarray(front_coverage).resize((r, r), Image.Resampling.BILINEAR)) > 0
    alpha = ref[..., 3] / 255.0 * sil
    front = ref[..., :3] * alpha[..., None] + 255.0 * (1 - alpha[..., None])

    init = np.full((6, r, r, 3), 255, np.uint8)
    init[0] = np.clip(front + 0.5, 0, 255).astype(np.uint8)
    inside = ndimage.distance_transform_edt(alpha > 0.5)
    keep = np.zeros((6, r, r), np.float32)
    keep[0] = np.clip((inside - erode_px) / feather_px, 0.0, 1.0)
    return init, keep


NODE_CLASS_MAPPINGS = {"Hy3dPinFront": Hy3dPinFront}
NODE_DISPLAY_NAME_MAPPINGS = {"Hy3dPinFront": "hy3d: Pin UltraTex front view to reference"}

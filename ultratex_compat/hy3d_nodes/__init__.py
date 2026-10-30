"""ComfyUI nodes used by ultratex_engine (installed into ComfyUI/custom_nodes/hy3d_nodes)."""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

FRONT, BACK = 0, 2  # UltraTex canonical view indices
# Pin fully where the surface faces the camera at least this much (cosine), nothing below FACING_LO:
# surfaces turning away are only a few pixels wide in the drawing, so pinning them stretches the art
# and smears the colours of whatever is drawn next to them onto the mesh.
FACING_LO, FACING_HI = 0.45, 0.75


def facing(normal_rgba, view):
    """Cosine between surface normal and the view direction, from UltraTex's normal-map encoding.

    rgb = (nx, ny, -nz)_blender * 0.5 + 0.5; the front camera looks along +y (Blender), the back along -y.
    """
    ny = normal_rgba[..., 1].astype(np.float32) / 127.5 - 1.0
    return (-ny if view == FRONT else ny) * (normal_rgba[..., 3] > 0)


class Hy3dPinFront:
    """Init views + keep masks that pin UltraTex's front (and optionally back) view to reference images.

    UltraTex conditions on the reference but repaints it, so logos, eyes and colours drift.
    Pinning views (RePaint keep mask on every step) makes the sampler paint the remaining views
    around the real art instead of its own reinterpretation of it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": ("ULTRATEX_PREP",),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
                "erode_px": ("INT", {"default": 6, "min": 0, "max": 256, "tooltip": "Unpinned band inside the outline (at the sampling resolution), where alignment is least reliable."}),
                "feather_px": ("INT", {"default": 12, "min": 1, "max": 256}),
            },
            "optional": {
                "back_image": ("IMAGE", {"tooltip": "Back view of the same object (LoadImage); pinned to UltraTex's back view."}),
                "back_mask": ("MASK", {"tooltip": "LoadImage's MASK for back_image (1 = transparent)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("init_views", "keep_mask")
    FUNCTION = "pin"
    CATEGORY = "hy3d"

    def pin(self, prep, resolution, erode_px, feather_px, back_image=None, back_mask=None):
        init, keep = pin_front(prep.reference_rgba, prep.normal_rgba[FRONT, ..., 3], resolution, erode_px, feather_px,
                               facing(prep.normal_rgba[FRONT], FRONT))
        if back_image is not None:
            rgb = np.clip(back_image[0].cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8)[..., :3]
            if back_mask is not None:
                m = back_mask[0].cpu().numpy()
                if m.shape != rgb.shape[:2]:
                    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(rgb.shape[1::-1])) / 255.0
                alpha = np.clip((1.0 - m) * 255 + 0.5, 0, 255).astype(np.uint8)
            else:
                alpha = np.full(rgb.shape[:2], 255, np.uint8)
            back_cov = prep.normal_rgba[BACK, ..., 3]
            aligned = fit_to_silhouette(np.dstack([rgb, alpha]), back_cov > 0, back_cov.shape[0])
            b_init, b_keep = pin_view(aligned, back_cov, resolution, erode_px, feather_px,
                                      facing(prep.normal_rgba[BACK], BACK))
            init[BACK], keep[BACK] = b_init, b_keep
        return (torch.from_numpy(init.astype(np.float32) / 255.0), torch.from_numpy(keep))


def solid_bbox(mask, min_frac=0.002):
    """Bounding box of the connected components that make up at least ``min_frac`` of ``mask``."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= max(1.0, min_frac * sizes.sum()))[0])
    ys, xs = np.nonzero(keep)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def fit_to_silhouette(rgba, silhouette, size):
    """Crop ``rgba`` to its solid alpha and fit it into the silhouette's bbox (UltraTex's front fit)."""
    box = solid_bbox(rgba[..., 3] >= 204)
    sbox = solid_bbox(silhouette)
    if box is None or sbox is None:
        raise ValueError("hy3d pin: empty back image or back silhouette")
    ref = Image.fromarray(rgba).crop(box)
    k = size / silhouette.shape[0]
    bx0, by0, bx1, by1 = (v * k for v in sbox)
    bw, bh = bx1 - bx0, by1 - by0
    s = min(bw / ref.width, bh / ref.height)
    w, h = max(1, round(ref.width * s)), max(1, round(ref.height * s))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(ref.resize((w, h), Image.Resampling.LANCZOS), (round(bx0 + (bw - w) / 2), round(by0 + (bh - h) / 2)))
    return np.asarray(canvas)


def pin_view(reference_rgba, coverage, resolution, erode_px=6, feather_px=12, cos=None):
    """One view's (R, R, 3) uint8 init image (reference over white) and (R, R) keep mask.

    ``cos`` (same size as ``coverage``): how directly each pixel faces the camera; the keep mask
    fades from FACING_HI to FACING_LO so surfaces turning away get repainted.
    """
    r = resolution
    ref = np.asarray(Image.fromarray(reference_rgba).resize((r, r), Image.Resampling.LANCZOS)).astype(np.float32)
    sil = np.asarray(Image.fromarray(coverage).resize((r, r), Image.Resampling.BILINEAR)) > 0
    alpha = ref[..., 3] / 255.0 * sil
    view = ref[..., :3] * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    inside = ndimage.distance_transform_edt(alpha > 0.5)
    keep = np.clip((inside - erode_px) / feather_px, 0.0, 1.0).astype(np.float32)
    if cos is not None:
        c = np.asarray(Image.fromarray(cos.astype(np.float32)).resize((r, r), Image.Resampling.BILINEAR))
        t = np.clip((c - FACING_LO) / (FACING_HI - FACING_LO), 0.0, 1.0)
        keep *= t * t * (3 - 2 * t)
    return np.clip(view + 0.5, 0, 255).astype(np.uint8), keep


def pin_front(reference_rgba, front_coverage, resolution, erode_px=6, feather_px=12, cos=None):
    """(6, R, R, 3) uint8 init views (white except the front) and (6, R, R) float keep masks."""
    r = resolution
    init = np.full((6, r, r, 3), 255, np.uint8)
    keep = np.zeros((6, r, r), np.float32)
    init[FRONT], keep[FRONT] = pin_view(reference_rgba, front_coverage, resolution, erode_px, feather_px, cos)
    return init, keep


NODE_CLASS_MAPPINGS = {"Hy3dPinFront": Hy3dPinFront}
NODE_DISPLAY_NAME_MAPPINGS = {"Hy3dPinFront": "hy3d: Pin UltraTex front/back views to reference images"}

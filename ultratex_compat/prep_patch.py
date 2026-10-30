"""Robust reference alignment for ComfyUI-UltraTex's prep (appended to ultratex_core/prep.py).

UltraTex fits the reference's alpha bounding box (alpha > 8) onto the front silhouette's
bounding box. Background removers leave faint alpha far from the object and meshes can have
tiny floaters, so both boxes come out too big and the reference lands shrunken and offset.
This version measures both boxes on solid pixels (alpha >= 128) and ignores specks.
"""

HOOK = '''

# --- hy3d: bounding boxes from solid, connected pixels only (see ultratex_compat/prep_patch.py)
def _hy3d_solid_bbox(mask, min_frac=0.002):
    from scipy import ndimage as _nd
    lab, n = _nd.label(mask)
    if n == 0:
        return None
    sizes = _nd.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= max(1.0, min_frac * sizes.sum()))[0])
    ys, xs = np.nonzero(keep)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


_hy3d_align_reference_orig = align_reference


def align_reference(rgba, front_alpha, size, mode):
    box = _hy3d_solid_bbox(rgba[..., 3] >= 128)
    if box is None:
        return _hy3d_align_reference_orig(rgba, front_alpha, size, mode)
    x0, y0, x1, y1 = box
    rgba = rgba.copy()
    outside = np.ones(rgba.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    rgba[outside] = 0  # faint alpha outside the object's box would otherwise widen the crop
    fbox = _hy3d_solid_bbox(np.asarray(front_alpha) > 0) if mode == "fit_front_silhouette" else None
    if fbox is not None:
        fx0, fy0, fx1, fy1 = fbox
        clean = np.zeros_like(front_alpha, dtype=bool)
        clean[fy0:fy1, fx0:fx1] = np.asarray(front_alpha)[fy0:fy1, fx0:fx1] > 0
        front_alpha = clean
    return _hy3d_align_reference_orig(rgba, front_alpha, size, mode)
'''

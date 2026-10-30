"""Which side of a generated mesh faces the camera that took the input image?

Image-to-3D models sculpt surface details (seams, cut-outs, belts, facial features) from the
input image, so the geometric creases on the correct side line up with the image's line art.
Compare edge maps of the mesh's normals rendered from +Z and from -Z with the image's edges,
ignoring a band around the outline (the silhouette looks the same from both sides).
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from project_front import foreground, rasterize


def _normal_render(verts, normals, faces, sign, size=512):
    x, y, z = sign * verts[:, 0], verts[:, 1], sign * verts[:, 2]
    s = (size - 8) / max(np.ptp(x), np.ptp(y))
    xy = np.stack([(x - x.min()) * s + 4, (y.max() - y) * s + 4], 1)
    face, px, py, bary = rasterize(xy[faces], size, size, box=12)
    order = np.argsort((z[faces[face]] * bary).sum(1))  # far to near: nearest surface wins
    img = np.zeros((size, size, 3))
    cov = np.zeros((size, size), bool)
    nn = (normals[faces[face[order]]] * bary[order][..., None]).sum(1) * np.array([sign, 1, sign])
    img[py[order], px[order]] = nn
    cov[py[order], px[order]] = True
    return img, cov


def _crop(mask, arr, out=(256, 512)):
    ys, xs = np.nonzero(mask)
    return np.asarray(Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32))
                      .resize(out, Image.BILINEAR))


def _edges(a):
    return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))


def _norm(e):
    e = ndimage.gaussian_filter(e, 3)
    return (e - e.mean()) / (e.std() + 1e-9)


def side_scores(verts, normals, faces, rgb, mask=None, band=8):
    """Edge correlation of the image with the mesh seen from +Z (key 1) and -Z (key -1)."""
    rgb = rgb.astype(np.float32)
    fg = foreground(rgb.astype(np.uint8)) if mask is None else mask
    inner = ndimage.binary_erosion(fg, iterations=band)
    ie = _norm(_crop(fg, _edges(rgb.mean(-1)) * inner))
    out = {}
    for sign in (1, -1):
        nr, cov = _normal_render(verts, normals, faces, sign)
        ginner = ndimage.binary_erosion(cov, iterations=max(2, band // 2))
        ge = _norm(_crop(cov, sum(_edges(nr[..., c]) for c in range(3)) * ginner))
        out[sign] = float((ie * ge).mean())
    return out

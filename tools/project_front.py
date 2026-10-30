#!/usr/bin/env python3
"""Project the source image onto the front of a generated GLB's base-color texture.

Image-to-3D models redraw the texture and lose fine line art (logos, eyes, lettering).
The input image already shows the front exactly, so this aligns the mesh's front
silhouette with the image's foreground and copies image pixels into every texel that
is visible from the front, fading out on surfaces that turn away and near the outline.
Back and occluded areas keep the generated texture.

    python3 tools/project_front.py result.glb source.png out.glb [--debug dir]

Runs on CPU (numpy/scipy/pygltflib/Pillow). Assumes glTF's +Y up and the model facing +Z,
as TRELLIS.2 and Hunyuan3D export, and a roughly orthographic, front-on source image.
"""

import argparse
import io
import struct
from pathlib import Path

import numpy as np
import pygltflib
from PIL import Image
from scipy import ndimage

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


# -- glTF io -------------------------------------------------------------------

def load(path):
    g = pygltflib.GLTF2().load(str(path))
    return g, g.binary_blob()


def view_bytes(g, blob, index):
    bv = g.bufferViews[index]
    return blob[bv.byteOffset or 0:(bv.byteOffset or 0) + bv.byteLength]


def accessor(g, blob, index):
    acc = g.accessors[index]
    bv = g.bufferViews[acc.bufferView]
    n = COMPONENTS[acc.type]
    dtype = np.dtype(DTYPES[acc.componentType])
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or dtype.itemsize * n
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (acc.count - 1) + dtype.itemsize * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(acc.count, dtype.itemsize * n), strides=(stride, 1))
    out = np.ascontiguousarray(rows).view(dtype).reshape(acc.count, n)
    return out[:, 0] if n == 1 else out


def image_index(g, texture_index):
    tex = g.textures[texture_index]
    if tex.source is not None:
        return tex.source
    for ext in ("EXT_texture_webp", "KHR_texture_basisu", "EXT_texture_avif"):
        if ext in (tex.extensions or {}):
            return tex.extensions[ext]["source"]
    raise ValueError("texture has no image source")


def replace_view(g, blob, index, data):
    """Return a new blob with bufferView ``index`` replaced by ``data`` (offsets re-packed)."""
    parts, offset = [], 0
    for i, bv in enumerate(g.bufferViews):
        chunk = data if i == index else view_bytes(g, blob, i)
        pad = (-offset) % 4
        parts.append(b"\0" * pad)
        offset += pad
        bv.byteOffset = offset
        bv.byteLength = len(chunk)
        parts.append(chunk)
        offset += len(chunk)
    new = b"".join(parts)
    g.buffers[0].byteLength = len(new)
    return new


# -- rasterization ---------------------------------------------------------------

def rasterize(tri, width, height, box=24, chunk=20000):
    """Cover pixel centers with 2D triangles (F,3,2 in pixel units).

    Returns (face, x, y, barycentrics[K,3]) for every covered pixel, possibly several per pixel.
    """
    lo = np.floor(tri.min(axis=1)).astype(np.int64)
    hi = np.ceil(tri.max(axis=1)).astype(np.int64)
    span = (hi - lo).max(axis=1)
    out = []

    def cover(fids, size):
        t = tri[fids]
        ox, oy = np.meshgrid(np.arange(size), np.arange(size))
        px = lo[fids, 0][:, None] + ox.ravel()[None] + 0.5
        py = lo[fids, 1][:, None] + oy.ravel()[None] + 0.5
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        den = np.where(np.abs(den) < 1e-12, np.nan, den)[:, None]
        w0 = ((b[:, 1] - c[:, 1])[:, None] * (px - c[:, 0][:, None]) + (c[:, 0] - b[:, 0])[:, None] * (py - c[:, 1][:, None])) / den
        w1 = ((c[:, 1] - a[:, 1])[:, None] * (px - c[:, 0][:, None]) + (a[:, 0] - c[:, 0])[:, None] * (py - c[:, 1][:, None])) / den
        w2 = 1 - w0 - w1
        eps = -1e-4
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (px < width) & (py < height) & (px >= 0) & (py >= 0)
        fi, k = np.nonzero(inside)
        out.append((fids[fi], (px[fi, k] - 0.5).astype(np.int64), (py[fi, k] - 0.5).astype(np.int64),
                    np.stack([w0[fi, k], w1[fi, k], w2[fi, k]], 1)))

    small = np.nonzero(span <= box)[0]
    for i in range(0, len(small), chunk):
        cover(small[i:i + chunk], box + 1)
    for f in np.nonzero(span > box)[0]:  # rare big triangles
        cover(np.array([f]), int(span[f]) + 1)
    if not out:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
    return tuple(np.concatenate(x) for x in zip(*out))


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = img.reshape(h, w, -1)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


# -- projection -------------------------------------------------------------------

def foreground(rgb, white=235):
    """Foreground mask: everything except near-white background connected to the border."""
    bgish = (rgb >= white).all(axis=2)
    labels, _ = ndimage.label(bgish)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg = np.isin(labels, border[border > 0])
    return ndimage.binary_fill_holes(~bg)


def silhouette(pos, size_hw, s, a, tx, ty):
    """Front silhouette by splatting vertices (dense meshes) then closing small gaps."""
    h, w = size_hw
    x = np.round(pos[:, 0] * s * a + tx).astype(int)
    y = np.round(-pos[:, 1] * s + ty).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    m = np.zeros((h, w), bool)
    m[y[ok], x[ok]] = True
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def align(pos, fg, work=512):
    """Fit image = (x*s*a + tx, -y*s + ty) so the mesh's front silhouette matches the foreground.

    Starts from matching bounding boxes, then coordinate descent on silhouette IoU at
    ``work`` px with shrinking steps. ``a`` allows a slight horizontal/vertical scale mismatch.
    """
    h, w = fg.shape
    ys, xs = np.nonzero(fg)
    ib = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    mb = np.array([pos[:, 0].min(), -pos[:, 1].max(), pos[:, 0].max(), -pos[:, 1].min()])
    sx = (ib[2] - ib[0]) / (mb[2] - mb[0])
    sy = (ib[3] - ib[1]) / (mb[3] - mb[1])
    params = {"s": sy, "a": sx / sy,
              "tx": (ib[0] + ib[2]) / 2 - sx * (mb[0] + mb[2]) / 2,
              "ty": (ib[1] + ib[3]) / 2 - sy * (mb[1] + mb[3]) / 2}

    k = work / max(h, w)
    fg_small = np.asarray(Image.fromarray(fg).resize((round(w * k), round(h * k)), Image.NEAREST))

    def iou(p):
        sil = silhouette(pos, fg_small.shape, p["s"] * k, p["a"], p["tx"] * k, p["ty"] * k)
        return (sil & fg_small).sum() / max(1, (sil | fg_small).sum())

    best = iou(params)
    for frac in (0.02, 0.01, 0.005, 0.0025, 0.001):
        steps = {"s": frac * params["s"], "a": frac, "tx": frac * w, "ty": frac * h}
        improved = True
        while improved:
            improved = False
            for key, step in steps.items():
                for sign in (1, -1):
                    trial = dict(params, **{key: params[key] + sign * step})
                    score = iou(trial)
                    if score > best + 1e-5:
                        best, params, improved = score, trial, True
    return best, params


def project(glb_in, image_in, glb_out, debug=None, face_lo=0.25, face_hi=0.6, edge_px=6, gutter_px=4):
    g, blob = load(glb_in)
    prim = g.meshes[0].primitives[0]
    pos = accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
    nrm = accessor(g, blob, prim.attributes.NORMAL).astype(np.float64)
    uv = accessor(g, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
    faces = accessor(g, blob, prim.indices).reshape(-1, 3).astype(np.int64)

    mat = g.materials[prim.material]
    img_idx = image_index(g, mat.pbrMetallicRoughness.baseColorTexture.index)
    image = g.images[img_idx]
    tex = Image.open(io.BytesIO(view_bytes(g, blob, image.bufferView)))
    tex_mode = tex.mode
    tex_rgb = np.asarray(tex.convert("RGB")).astype(np.float32)
    th, tw = tex_rgb.shape[:2]

    src = np.asarray(Image.open(image_in).convert("RGB")).astype(np.float32)
    h, w = src.shape[:2]
    fg = foreground(src.astype(np.uint8))

    iou, fit = align(pos, fg)
    s, a, tx, ty = fit["s"], fit["a"], fit["tx"], fit["ty"]
    print(f"alignment: silhouette IoU {iou:.3f}, scale {s:.1f}px/unit, aspect {a:.3f}, offset ({tx:.1f}, {ty:.1f})")
    img_xy = np.stack([pos[:, 0] * s * a + tx, -pos[:, 1] * s + ty], 1)

    # Front depth buffer (+Z faces the viewer, larger z is closer).
    face, x, y, bary = rasterize(img_xy[faces], w, h, box=16)
    z = (pos[faces[face], 2] * bary).sum(1)
    zbuf = np.full((h, w), -np.inf)
    np.maximum.at(zbuf, (y, x), z)
    covered = np.isfinite(zbuf)

    # Every texel: 3D position, normal, and where it lands in the image.
    tri_uv = np.stack([uv[:, 0] * tw, uv[:, 1] * th], 1)[faces]
    face, x, y, bary = rasterize(tri_uv, tw, th)
    p = (pos[faces[face]] * bary[..., None]).sum(1)
    n = (nrm[faces[face]] * bary[..., None]).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ix, iy = p[:, 0] * s * a + tx, -p[:, 1] * s + ty

    xi = np.clip(np.round(ix).astype(int), 0, w - 1)
    yi = np.clip(np.round(iy).astype(int), 0, h - 1)
    tol = 0.006 * (pos[:, 2].max() - pos[:, 2].min()) + 1.5 / s
    # Max over a 3x3 neighborhood so texels on silhouette edges aren't culled by their own neighbors.
    zmax = ndimage.maximum_filter(np.where(covered, zbuf, -np.inf), size=3)
    visible = p[:, 2] >= zmax[yi, xi] - tol
    facing = np.clip((n[:, 2] - face_lo) / (face_hi - face_lo), 0, 1)
    facing = facing * facing * (3 - 2 * facing)
    inside = ndimage.distance_transform_edt(fg)
    edge = np.clip(bilinear(inside.astype(np.float32), ix, iy)[:, 0] / edge_px, 0, 1)
    weight = (visible * facing * edge).astype(np.float32)[:, None]

    color = bilinear(src, ix, iy)
    # Blend as a delta so it can be spread into the atlas gutters: texels between UV charts
    # are covered by no triangle, and leaving them at the old color shows up as cracks
    # once the texture is filtered.
    delta = np.zeros_like(tex_rgb)
    has = np.zeros((th, tw), bool)
    delta[y, x] = (color - tex_rgb[y, x]) * weight
    has[y, x] = True
    dist, (ny, nx) = ndimage.distance_transform_edt(~has, return_indices=True)
    spread = (~has) & (dist <= gutter_px)
    delta[spread] = delta[ny[spread], nx[spread]]
    out = tex_rgb + delta
    changed = (weight[:, 0] > 0.01).sum()
    print(f"texels from source image: {changed} ({100 * changed / (tw * th):.1f}% of the atlas)")

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if tex_mode == "RGBA":
        result.putalpha(tex.getchannel("A"))
    buf = io.BytesIO()
    fmt = {"image/webp": "WEBP", "image/png": "PNG", "image/jpeg": "JPEG"}[image.mimeType]
    result.save(buf, format=fmt, **({"quality": 95} if fmt != "PNG" else {}))
    new_blob = replace_view(g, blob, image.bufferView, buf.getvalue())
    g.set_binary_blob(new_blob)
    g.save_binary(str(glb_out))

    if debug:
        d = Path(debug)
        d.mkdir(parents=True, exist_ok=True)
        wmap = np.zeros((th, tw), np.float32)
        wmap[y, x] = weight[:, 0]
        Image.fromarray((wmap * 255).astype(np.uint8)).resize((1024, 1024)).save(d / "weight_atlas.png")
        overlay = src.copy()
        sil = covered
        overlay[sil & ~fg] = [255, 0, 0]
        overlay[fg & ~sil] = [0, 0, 255]
        Image.fromarray(overlay.astype(np.uint8)).save(d / "alignment.png")
    return iou


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("glb")
    p.add_argument("image")
    p.add_argument("out")
    p.add_argument("--debug", help="write alignment and weight maps here")
    p.add_argument("--edge", type=float, default=6, help="fade width in image pixels at the outline")
    a = p.parse_args()
    project(a.glb, a.image, a.out, a.debug, edge_px=a.edge)


if __name__ == "__main__":
    main()

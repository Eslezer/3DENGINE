#!/usr/bin/env python3
"""Texture a GLB from front (and optionally back) images alone: no AI painter.

    python3 tools/project_views.py model.glb front.png out.glb [--back back.png] [--debug dir]

1. Each image is aligned to the mesh's silhouette from its side (silhouette IoU fit, as in
   project_front.py; the back camera sees the model mirrored in x).
2. Every texel the camera sees (depth-tested) takes the image's colour, weighted by how
   directly the surface faces that camera and faded near the outline.
3. Surfaces neither image covers well (sides, top of the shoulders, soles) are filled by a
   harmonic interpolation over the mesh surface from the projected colours: smooth, and
   continuous across UV seams because duplicated seam vertices are welded for the solve.

Deterministic, CPU only, and nothing is generated: every colour comes from the images.
Assumes +Y up, the model facing +Z, and front-on, roughly orthographic drawings on a plain
light background (see project_front.foreground).
"""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import factorized

from project_front import (accessor, align, bilinear, foreground, image_index, load, rasterize,
                           replace_view, view_bytes)

FACE_LO, FACE_HI = 0.15, 0.55  # facing cosine where a view starts / fully counts
SEED_WEIGHT = 0.5  # vertices at least this well covered seed the side fill


class View:
    """One source image, aligned to the mesh as seen from the front (+Z) or back (-Z)."""

    def __init__(self, path, pos, faces, back=False, edge_px=6):
        self.back = back
        self.src = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
        h, w = self.src.shape[:2]
        self.fg = foreground(self.src.astype(np.uint8))
        self.sign = -1.0 if back else 1.0  # back camera: image x runs along -X, depth along -Z
        mirrored = pos * np.array([self.sign, 1, 1])
        self.iou, fit = align(mirrored, self.fg)
        self.s, self.a, self.tx, self.ty = fit["s"], fit["a"], fit["tx"], fit["ty"]
        print(f"{'back' if back else 'front'}: silhouette IoU {self.iou:.3f}, scale {self.s:.1f}px/unit, "
              f"aspect {self.a:.3f}")

        xy = self.to_image(pos)
        face, x, y, bary = rasterize(xy[faces], w, h, box=16)
        depth = (self.sign * pos[faces[face], 2] * bary).sum(1)  # larger = closer to this camera
        zbuf = np.full((h, w), -np.inf)
        np.maximum.at(zbuf, (y, x), depth)
        # Max over 3x3 so texels on silhouette edges are not culled by their own neighbours.
        self.zmax = ndimage.maximum_filter(zbuf, size=3)
        self.tol = 0.006 * np.ptp(pos[:, 2]) + 1.5 / self.s
        self.inside = ndimage.distance_transform_edt(self.fg).astype(np.float32)
        self.edge_px = edge_px

    def to_image(self, p):
        return np.stack([self.sign * p[:, 0] * self.s * self.a + self.tx, -p[:, 1] * self.s + self.ty], 1)

    def sample(self, p, n):
        """Colour and weight (0..1) of this view for surface points ``p`` with normals ``n``."""
        h, w = self.fg.shape
        xy = self.to_image(p)
        xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        visible = self.sign * p[:, 2] >= self.zmax[yi, xi] - self.tol
        t = np.clip((self.sign * n[:, 2] - FACE_LO) / (FACE_HI - FACE_LO), 0, 1)
        edge = np.clip(bilinear(self.inside, xy[:, 0], xy[:, 1])[:, 0] / self.edge_px, 0, 1)
        weight = visible * t * t * (3 - 2 * t) * edge
        return bilinear(self.src, xy[:, 0], xy[:, 1]), weight.astype(np.float32)


def combine(views, p, n):
    """Weighted mix of all views' colours; returns (colour, total weight clipped to 1)."""
    acc = np.zeros((len(p), 3), np.float32)
    wsum = np.zeros(len(p), np.float32)
    for v in views:
        c, w = v.sample(p, n)
        acc += c * w[:, None]
        wsum += w
    color = acc / np.maximum(wsum, 1e-6)[:, None]
    return color, np.clip(wsum, 0, 1)


def harmonic_fill(pos, faces, color, known):
    """Smoothly extend ``color`` from ``known`` vertices over the rest of the surface.

    Seam duplicates are welded so the fill crosses UV seams. Components with no known
    vertex are returned as NaN.
    """
    key = np.round(pos / (np.ptp(pos, 0).max() * 1e-5)).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    m = len(weld)
    f = inverse[faces]
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = edges[edges[:, 0] != edges[:, 1]]
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(m, m)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)
    lap = sparse.diags(np.asarray(adj.sum(1)).ravel()) - adj

    # Average known colours onto welded vertices.
    k_w = np.zeros(m, bool)
    c_w = np.zeros((m, 3))
    cnt = np.zeros(m)
    np.add.at(c_w, inverse[known], color[known])
    np.add.at(cnt, inverse[known], 1)
    k_w[cnt > 0] = True
    c_w[k_w] /= cnt[k_w, None]

    # Only solve inside components that contain a known vertex (otherwise the system is singular).
    _, comp = sparse.csgraph.connected_components(adj, directed=False)
    seeded = np.isin(comp, np.unique(comp[k_w]))
    unknown = np.nonzero(seeded & ~k_w)[0]
    out = np.full((m, 3), np.nan)
    out[k_w] = c_w[k_w]
    if len(unknown):
        luu = lap[unknown][:, unknown].tocsc()
        luk = lap[unknown][:, np.nonzero(k_w)[0]]
        solve = factorized(luu)
        rhs = -(luk @ c_w[k_w])
        out[unknown] = np.stack([solve(rhs[:, c]) for c in range(3)], 1)
    return out[inverse]


def project_views(glb_in, front, glb_out, back=None, debug=None, edge_px=6, gutter_px=4):
    g, blob = load(glb_in)
    prim = g.meshes[0].primitives[0]
    pos = accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
    nrm = accessor(g, blob, prim.attributes.NORMAL).astype(np.float64)
    uv = accessor(g, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
    faces = accessor(g, blob, prim.indices).reshape(-1, 3).astype(np.int64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    mat = g.materials[prim.material]
    image = g.images[image_index(g, mat.pbrMetallicRoughness.baseColorTexture.index)]
    tex = Image.open(io.BytesIO(view_bytes(g, blob, image.bufferView)))
    base = np.asarray(tex.convert("RGB")).astype(np.float32)
    th, tw = base.shape[:2]

    views = [View(front, pos, faces, back=False, edge_px=edge_px)]
    if back:
        views.append(View(back, pos, faces, back=True, edge_px=edge_px))

    # Side fill: per-vertex colours from the views, extended over the surface.
    vcol, vw = combine(views, pos, nrm)
    known = vw >= SEED_WEIGHT
    print(f"fill seeds: {100 * known.mean():.0f}% of vertices covered by the images")
    fill = harmonic_fill(pos, faces, vcol, known)
    unfilled = np.isnan(fill[:, 0])

    # Per texel: image colour where the views see it, the fill elsewhere, the old texture as a last resort.
    tri_uv = np.stack([uv[:, 0] * tw, uv[:, 1] * th], 1)[faces]
    face, x, y, bary = rasterize(tri_uv, tw, th)
    fv = faces[face]
    p = (pos[fv] * bary[..., None]).sum(1)
    n = (nrm[fv] * bary[..., None]).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    color, w = combine(views, p, n)
    fill_t = np.nan_to_num((fill[fv] * bary[..., None]).sum(1))
    has_fill = ~unfilled[fv].any(1)
    fill_t[~has_fill] = base[y[~has_fill], x[~has_fill]]
    texel = color * w[:, None] + fill_t * (1 - w[:, None])

    out = base.copy()
    has = np.zeros((th, tw), bool)
    out[y, x] = texel
    has[y, x] = True
    # Extend into the gutters between UV charts so filtering doesn't show cracks.
    dist, (ny, nx) = ndimage.distance_transform_edt(~has, return_indices=True)
    spread = (~has) & (dist <= gutter_px)
    out[spread] = out[ny[spread], nx[spread]]
    print(f"texels from images: {100 * (w > 0.5).mean():.0f}%, filled: {100 * (w <= 0.5).mean():.0f}% "
          f"(old texture kept on {100 * (~has_fill).mean():.1f}%)")

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if tex.mode == "RGBA":
        result.putalpha(tex.getchannel("A"))
    buf = io.BytesIO()
    fmt = {"image/webp": "WEBP", "image/png": "PNG", "image/jpeg": "JPEG"}[image.mimeType]
    result.save(buf, format=fmt, **({"quality": 95} if fmt != "PNG" else {}))
    g.set_binary_blob(replace_view(g, blob, image.bufferView, buf.getvalue()))
    g.save_binary(str(glb_out))

    if debug:
        d = Path(debug)
        d.mkdir(parents=True, exist_ok=True)
        wmap = np.zeros((th, tw), np.float32)
        wmap[y, x] = w
        Image.fromarray((wmap * 255).astype(np.uint8)).resize((1024, 1024)).save(d / "image_weight_atlas.png")
    return [v.iou for v in views]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("glb")
    p.add_argument("front")
    p.add_argument("out")
    p.add_argument("--back")
    p.add_argument("--debug")
    p.add_argument("--edge", type=float, default=6, help="fade width in image pixels at the outline")
    a = p.parse_args()
    project_views(a.glb, a.front, a.out, a.back, a.debug, edge_px=a.edge)


if __name__ == "__main__":
    main()

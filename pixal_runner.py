#!/usr/bin/env python3
"""Run Pixal3D (TencentARC, MIT) on one front image, or on front + back views; runs in the TRELLIS.2 env.

    python pixal_runner.py --repo <Pixal3D checkout> --front front.png [--back back.png] --out out.glb
        [--resolution 1536] [--seed 42] [--fov 0.2] [--faces 500000] [--texture 4096]

Pixal3D is built on TRELLIS.2 and shares its extensions, so it runs in the same Python as
trellis_engine, but as a subprocess so its models never sit next to TRELLIS.2's in VRAM.
Changes to the stock scripts (applied by patching at import time, not by editing them):
  * attention goes through trellis_compat's xformers stand-in (torch SDPA), no flash-attn;
  * natten (needed by the NAF feature upsampler) is installed without its CUDA library, and
    NAF's neighbourhood attention runs on na2d_torch below instead of natten's cutlass kernel
    (natten's flex fallback builds a dense mask: 512 GiB at NAF's 1024 x 1024);
  * the gated RMBG-2.0 background remover is swapped for the open BiRefNet;
  * the GLB is exported with PNG textures and our face/texture limits.

With --back, both drawings are framed into a front (azimuth 0) and back (azimuth 180) camera
rig and fed to Pixal3D's multi-view pipeline, so the back is reconstructed from the drawing
instead of guessed. Images without real transparency are cut out with BiRefNet first.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
FRAME_FILL = 1 / 1.1  # object size / frame size, as in Pixal3D's preprocess_image


def na2d_torch(query, key, value, kernel_size, dilation=1, stride=1, scale=None, chunk_bytes=2 << 30, **_):
    """2-D neighbourhood attention in plain PyTorch, matching natten.na2d's semantics.

    Layout [B, X, Y, heads, dim] (V may have its own dim). Each query attends to the K x K window
    centred on it, shifted inwards at the borders so it always has K x K neighbours; dilation d
    splits the grid into d x d interleaved sub-grids that attend independently. Processed in row
    chunks, so memory stays small even for 1024 x 1024 maps (NAF's upsampler).
    """
    kx, ky = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    dx, dy = (dilation, dilation) if isinstance(dilation, int) else dilation
    if (stride if isinstance(stride, int) else max(stride)) != 1:
        raise NotImplementedError("na2d_torch: stride must be 1")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    if dx > 1 or dy > 1:
        out = query.new_empty(*query.shape[:-1], value.shape[-1])
        for i in range(dx):
            for j in range(dy):
                out[:, i::dx, j::dy] = na2d_torch(query[:, i::dx, j::dy], key[:, i::dx, j::dy],
                                                  value[:, i::dx, j::dy], (kx, ky), 1, 1, scale, chunk_bytes)
        return out
    b, X, Y, h, _ = query.shape
    if X < kx or Y < ky:
        raise ValueError(f"na2d_torch: map {X}x{Y} smaller than kernel {kx}x{ky}")
    dev = query.device
    rows = (torch.arange(X, device=dev) - kx // 2).clamp(0, X - kx)[:, None] + torch.arange(kx, device=dev)  # [X, kx]
    cols = (torch.arange(Y, device=dev) - ky // 2).clamp(0, Y - ky)[:, None] + torch.arange(ky, device=dev)  # [Y, ky]
    per_row = b * kx * Y * ky * h * (query.shape[-1] + value.shape[-1]) * query.element_size()
    step = max(1, int(chunk_bytes // max(per_row, 1)))
    out = []
    for r0 in range(0, X, step):
        r = rows[r0:r0 + step]                                    # [R, kx]
        kr, vr = key[:, r], value[:, r]                           # [B, R, kx, Y, h, d]
        kg, vg = kr[:, :, :, cols], vr[:, :, :, cols]             # [B, R, kx, Y, ky, h, d]
        q = query[:, r0:r0 + step]                                # [B, R, Y, h, d]
        logits = torch.einsum("brwhd,brawchd->brwhac", q.float(), kg.float()) * scale
        w = logits.flatten(-2).softmax(-1).view_as(logits).to(value.dtype)
        out.append(torch.einsum("brwhac,brawchd->brwhd", w, vg))
    return torch.cat(out, 1)


def patch_natten():
    """NAF calls natten's compiled cutlass kernel, which a pure-Python natten install lacks; use na2d_torch."""
    import natten

    def na2d(query, key, value, kernel_size, *args, stride=1, dilation=1, scale=None, backend=None, **kw):
        return na2d_torch(query, key, value, kernel_size, dilation, stride, scale)

    natten.na2d = na2d  # NAF does `from natten import na2d` when torch.hub imports it


def patch_rembg():
    from pixal3d.pipelines import rembg
    sys.path.insert(0, str(ROOT))
    from trellis_engine import use_open_rembg
    use_open_rembg(rembg.BiRefNet)


def cut_out(paths, dst_dir):
    """RGBA cutouts of ``paths`` (BiRefNet unless already transparent) saved into ``dst_dir``."""
    from hy3d_queue import has_transparency
    from pixal3d.pipelines.rembg import BiRefNet

    model, out = None, []
    for i, path in enumerate(paths):
        image = Image.open(path)
        if not has_transparency(image):
            if model is None:
                model = BiRefNet("ZhengPeng7/BiRefNet")
                model.cuda()
            image = model(image.convert("RGB"))
        image = image.convert("RGBA")
        image.putalpha(image.getchannel("A").point(lambda v: 0 if v < 32 else v))  # drop faint haze
        dst = Path(dst_dir) / f"cutout_{i}.png"
        image.save(dst)
        out.append(str(dst))
    if model is not None:
        model.cpu()
        del model
    return out


def capture_glb(faces, texture):
    """Make Pixal3D's to_glb use our limits and remember the mesh it returns (after its rotation)."""
    import o_voxel
    captured = []
    orig = o_voxel.postprocess.to_glb

    def to_glb(*args, **kw):
        kw["decimation_target"] = faces
        kw["texture_size"] = texture
        mesh = orig(*args, **kw)
        captured.append(mesh)
        return mesh

    o_voxel.postprocess.to_glb = to_glb
    return captured


def frame(rgba, size, height_frac):
    """Crop to the solid alpha, scale so the object's height is ``height_frac`` of ``size``, centre."""
    a = np.asarray(rgba)[..., 3]
    ys, xs = np.nonzero(a > 204)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    obj = rgba.crop(box)
    s = height_frac * size / obj.height
    obj = obj.resize((max(1, round(obj.width * s)), max(1, round(obj.height * s))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
    return canvas


def mv_views(repo, front, back, work):
    """Views dir with the front and back drawings on Pixal3D's azimuth 0 / 180 cameras."""
    rig = json.loads((Path(repo) / "assets/mv_images/example/transforms.json").read_text())
    frames = {f["name"]: f for f in rig["frames"]}
    f_img, b_img = Image.open(front).convert("RGBA"), Image.open(back).convert("RGBA")
    # Same scale for both: the taller drawing's object (normally the same character) fills FRAME_FILL.
    fa, ba = (np.asarray(i)[..., 3] > 204 for i in (f_img, b_img))
    fh = np.ptp(np.nonzero(fa)[0]) / max(np.ptp(np.nonzero(fa)[1]), np.ptp(np.nonzero(fa)[0]))
    bh = np.ptp(np.nonzero(ba)[0]) / max(np.ptp(np.nonzero(ba)[1]), np.ptp(np.nonzero(ba)[0]))
    d = Path(work) / "views"
    d.mkdir(parents=True, exist_ok=True)
    frame(f_img, 1024, FRAME_FILL * fh).save(d / "front.png")
    frame(b_img, 1024, FRAME_FILL * bh).save(d / "back.png")
    out = {"camera_angle_x": rig["camera_angle_x"], "mesh_scale": rig.get("mesh_scale", 1.0), "frames": [
        dict(frames["azim000"], file_path="front.png"), dict(frames["azim180"], file_path="back.png")]}
    (d / "transforms.json").write_text(json.dumps(out, indent=1))
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--front", required=True)
    p.add_argument("--back")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=1536, choices=[1024, 1536])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fov", type=float, default=0.2,
                   help="single view: camera FOV in radians (drawings are near-orthographic); <=0 = estimate with MoGe")
    p.add_argument("--faces", type=int, default=500_000)
    p.add_argument("--texture", type=int, default=4096)
    p.add_argument("--cutouts", help="directory to save the front/back cutouts in (default: a temp dir)")
    a = p.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "xformers")  # served by trellis_compat/xformers (torch SDPA)
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo = Path(a.repo).resolve()
    a.front, a.back = str(Path(a.front).resolve()), a.back and str(Path(a.back).resolve())
    a.out = str(Path(a.out).resolve())
    for path in (str(ROOT), str(repo), str(ROOT / "trellis_compat")):
        sys.path.insert(0, path)
    os.chdir(repo)  # the scripts keep their autotune cache and example assets next to themselves

    patch_natten()
    patch_rembg()
    captured = capture_glb(a.faces, a.texture)
    work = Path(tempfile.mkdtemp(prefix="pixal_"))
    scratch = str(work / "pixal_raw.glb")
    cuts = cut_out([a.front] + ([a.back] if a.back else []), a.cutouts or work)
    if a.back:
        import inference_mv
        views = mv_views(repo, cuts[0], cuts[1], work)
        inference_mv.run_inference(views_dir=str(views), output_path=scratch, seed=a.seed, resolution=a.resolution)
    else:
        import inference
        inference.run_inference(image_path=cuts[0], output_path=scratch, seed=a.seed,
                                manual_fov=a.fov, resolution=a.resolution)
    mesh = captured[-1]
    mesh.export(a.out)
    print(f"pixal3d: wrote {a.out} ({len(mesh.faces)} faces, {'front+back' if a.back else 'front only'})", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Pixal3D (TencentARC, MIT) on one front image, or on front + back views; runs in the TRELLIS.2 env.

    python pixal_runner.py --repo <Pixal3D checkout> --front front.png [--back back.png] --out out.glb
        [--resolution 1536] [--seed 42] [--fov 0.2] [--faces 500000] [--texture 4096]

Pixal3D is built on TRELLIS.2 and shares its extensions, so it runs in the same Python as
trellis_engine, but as a subprocess so its models never sit next to TRELLIS.2's in VRAM.
Changes to the stock scripts (applied by patching at import time, not by editing them):
  * attention goes through trellis_compat's xformers stand-in (torch SDPA), no flash-attn;
  * natten (needed by the NAF feature upsampler) is installed without its CUDA library and
    runs on PyTorch's compiled flex attention instead of the cutlass kernels NAF asks for;
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
from PIL import Image

ROOT = Path(__file__).resolve().parent
FRAME_FILL = 1 / 1.1  # object size / frame size, as in Pixal3D's preprocess_image


def patch_natten():
    import natten
    import torch

    natten.allow_flex_compile()
    orig = natten.na2d

    compiled = [True]

    def flex(q, k, v, **kw):
        if compiled[0]:
            try:
                return orig(q, k, v, backend="flex-fna", torch_compile=True, **kw)
            except Exception as e:  # e.g. no C compiler for Triton; eager flex is slower and hungrier but exact
                print(f"natten: compiled flex failed ({type(e).__name__}: {e}); using eager flex", flush=True)
                compiled[0] = False
        return orig(q, k, v, backend="flex-fna", torch_compile=False, **kw)

    def na2d(query, key, value, *args, backend=None, torch_compile=False, **kw):
        if args:
            kw["kernel_size"] = args[0]
        d = query.shape[-1]
        if value.shape[-1] == d:
            return flex(query, key, value, **kw)
        # Flex needs equal QK and V head dims (NAF uses 64 and 256). The attention weights depend
        # only on Q and K, so attending to V in d-wide slices and concatenating is exact.
        pad = (-value.shape[-1]) % d
        v = torch.nn.functional.pad(value, (0, pad)) if pad else value
        out = torch.cat([flex(query, key, c.contiguous(), **kw) for c in v.split(d, dim=-1)], dim=-1)
        return out[..., :value.shape[-1]]

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

#!/usr/bin/env python3
"""Paint a mesh with Hunyuan3D-2.1's PBR texture model; runs inside the separate hy21 venv.

    python hy21_runner.py --repo <Hunyuan3D-2.1 checkout> --mesh in.glb --image front.png \\
        [--image back.png] --out out.glb [--faces 150000] [--views 8] [--view-size 768]

Hunyuan3D-2.1 pins libraries that clash with TRELLIS.2's (diffusers 0.30, transformers 4.46),
so hy21_engine.py installs it in its own venv and runs this script as a subprocess. Changes
to the stock pipeline (see its textureGenPipeline.py):
  * bpy (Blender, Python 3.10 only) is stubbed: it is only used to convert OBJ to GLB,
    which is done here with trimesh, keeping metallic and roughness.
  * Real-ESRGAN upscaling runs through spandrel instead of realesrgan/basicsr, which do not
    build on Python 3.13.
  * The mesh is decimated (fast_simplification) to --faces (default 150k) instead of the stock 40k remesh.

Hunyuan3D-2.1 is under the Tencent Hunyuan Non-Commercial License (not valid in the EU, UK or
South Korea).
"""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


def install_shims(esrgan_path):
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))  # only used by the OBJ->GLB converter

    import torch
    import utils.image_super_utils as sup

    class SpandrelSuper:
        """Drop-in for imageSuperNet: x4 Real-ESRGAN via spandrel."""

        def __init__(self, config):
            from spandrel import ModelLoader
            self.model = ModelLoader().load_from_file(esrgan_path).cuda().eval().half()

        @torch.no_grad()
        def __call__(self, image):
            x = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1)[None].cuda().half() / 255
            y = self.model(x).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
            return Image.fromarray((y * 255 + 0.5).astype(np.uint8))

    sup.imageSuperNet = SpandrelSuper


def patch_pipeline(path):
    """Upstream __call__ leaves image_prompt unset when given a list of images (front + back)."""
    text = path.read_text()
    old = "elif isinstance(image_path, Image.Image):"
    if old in text:
        path.write_text(text.replace(old, "elif isinstance(image_path, (Image.Image, list)):"))


def decimate(src, dst, faces):
    """Weld UV seams and quadric-decimate to ``faces``, written as a plain OBJ (no UVs/material)."""
    import fast_simplification
    import trimesh

    mesh = trimesh.load(src, force="mesh", process=False)
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)  # welds duplicated seam vertices
    if len(mesh.faces) > faces:
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                            target_reduction=1 - faces / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=True)
    mesh.export(dst)


def obj_to_glb(obj_path, glb_path):
    """OBJ + MTL (map_Kd, map_Pm, map_Pr) -> GLB with a glTF metallic-roughness material."""
    import trimesh

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    folder = Path(obj_path).parent
    maps = {}
    for line in (folder / (Path(obj_path).stem + ".mtl")).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("map_Kd", "map_Pm", "map_Pr"):
            maps[parts[0]] = folder / parts[-1]
    albedo = Image.open(maps["map_Kd"]).convert("RGB")
    orm = None
    if "map_Pm" in maps and "map_Pr" in maps:
        metal = np.asarray(Image.open(maps["map_Pm"]).convert("L").resize(albedo.size))
        rough = np.asarray(Image.open(maps["map_Pr"]).convert("L").resize(albedo.size))
        orm = Image.fromarray(np.dstack([np.full_like(metal, 255), rough, metal]))  # R occlusion, G rough, B metal
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=albedo, metallicRoughnessTexture=orm,
        metallicFactor=1.0 if orm is not None else 0.0, roughnessFactor=1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
    mesh.export(glb_path)
    return mesh


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--image", action="append", required=True, help="reference image(s); front first")
    p.add_argument("--out", required=True)
    p.add_argument("--faces", type=int, default=150_000)
    p.add_argument("--views", type=int, default=8, help="generated views, 6 to 9")
    p.add_argument("--view-size", type=int, default=768, choices=[512, 768])
    p.add_argument("--work", default=None)
    a = p.parse_args()

    paint_dir = Path(a.repo) / "hy3dpaint"
    sys.path.insert(0, str(paint_dir))
    os.chdir(paint_dir)  # the pipeline resolves some paths relative to hy3dpaint/
    esrgan = paint_dir / "ckpt" / "RealESRGAN_x4plus.pth"
    install_shims(str(esrgan))
    patch_pipeline(paint_dir / "textureGenPipeline.py")

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    work = Path(a.work or Path(a.out).parent / "hy21_work")
    work.mkdir(parents=True, exist_ok=True)
    mesh_in = work / "mesh.obj"
    decimate(a.mesh, mesh_in, a.faces)

    conf = Hunyuan3DPaintConfig(a.views, a.view_size)
    conf.multiview_cfg_path = str(paint_dir / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(esrgan)
    pipe = Hunyuan3DPaintPipeline(conf)
    images = [Image.open(i).convert("RGBA") for i in a.image]
    obj = pipe(mesh_path=str(mesh_in), image_path=images, output_mesh_path=str(work / "textured.obj"),
               use_remesh=False, save_glb=False)
    mesh = obj_to_glb(obj, a.out)
    print(f"hy21: wrote {a.out} ({len(mesh.faces)} faces)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sharpen any textured model with the user's own drawings: the "mix" recipe.

    python3 tools/remix.py model.(glb|obj) front.png out_dir [--back back.png] [--no-upscale]

1. Loads the model (GLB, or OBJ + MTL + texture, e.g. from another tool the user sent).
2. Upscales its base-colour texture 4x with anime Real-ESRGAN, then to --texture px (4096).
3. Upscales the drawings 4x (capped at --max-side, 4096 by default).
4. Projects front/back drawings onto the surfaces facing those cameras and keeps the model's
   own texture everywhere else (tools/project_views.py --keep-base).
5. Writes out_dir/<name>.glb and out_dir/<name>_obj.zip (OBJ + MTL + PNG).

Assumes +Y up and the model facing +Z (TRELLIS.2/Hunyuan outputs after our orientation
fix, and the user's other-tool OBJ). Check the printed silhouette IoU: below ~0.85 the art
lands off-target. CPU only; at 4K expect ~10 min for the projection step.
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_views import project_views  # noqa: E402


def base_texture(mesh):
    mat = mesh.visual.material
    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if tex is None:
        raise SystemExit("model has no base-colour texture")
    return tex


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("front")
    p.add_argument("out_dir")
    p.add_argument("--back")
    p.add_argument("--name", default="remix")
    p.add_argument("--texture", type=int, default=4096)
    p.add_argument("--max-side", type=int, default=4096, help="cap for the upscaled drawings")
    p.add_argument("--no-upscale", action="store_true", help="skip Real-ESRGAN (no torch needed)")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="remix_"))

    mesh = trimesh.load(a.model, force="mesh")
    tex = base_texture(mesh).convert("RGB")
    drawings = [a.front] + ([a.back] if a.back else [])
    if not a.no_upscale:
        from upscale import upscale
        print(f"upscaling texture {tex.size}", flush=True)
        tex = upscale(tex)
        sharp = []
        for i, d in enumerate(drawings):
            print(f"upscaling {d}", flush=True)
            dst = work / f"drawing{i}.png"
            upscale(Image.open(d), max_side=a.max_side).convert("RGB").save(dst)
            sharp.append(str(dst))
        drawings = sharp
    tex = tex.resize((a.texture, a.texture), Image.Resampling.LANCZOS)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0))
    src = work / "base.glb"
    mesh.export(src)

    glb = out / f"{a.name}.glb"
    edge = 6 * max(1, Image.open(drawings[0]).height // 1024)  # fade width scales with the drawing size
    project_views(src, drawings[0], glb, drawings[1] if len(drawings) > 1 else None, edge_px=edge, keep_base=True)

    m = trimesh.load(glb, force="mesh")
    image = base_texture(m).convert("RGB")
    m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv, material=trimesh.visual.material.SimpleMaterial(
        image=image, diffuse=(255, 255, 255, 255)))
    obj_dir = work / "obj"
    obj_dir.mkdir()
    m.export(obj_dir / f"{a.name}.obj")
    with zipfile.ZipFile(out / f"{a.name}_obj.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in obj_dir.iterdir():
            z.write(f, f.name)
    print(f"wrote {glb} and {out / f'{a.name}_obj.zip'}", flush=True)


if __name__ == "__main__":
    main()

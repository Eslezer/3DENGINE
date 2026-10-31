"""Hunyuan3D-2.1 PBR painter for TRELLIS.2 meshes (painter=hy21).

Hunyuan3D-2.1's texture model generates albedo plus metallic/roughness views from the
reference image(s) and bakes them, with lighting taken out of the albedo. It pins
diffusers 0.30 and transformers 4.46, which clash with TRELLIS.2, so it lives in its own
venv (Python 3.13, made with uv) and runs as a subprocess (hy21_runner.py).

Its CUDA rasterizer and mesh-inpaint extension are prebuilt on the ``wheels`` branch
(hy21/sm120/cp313); the inpaint extension is compiled locally when missing.

Weights (tencent/Hunyuan3D-2.1, facebook/dinov2-giant) download on the first job, into the
shared Hugging Face cache. Licence: Tencent Hunyuan Non-Commercial (not valid in the EU, UK
or South Korea).
"""

import base64
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"
COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
PYTHON = "3.13"
TORCH = ["torch==2.11.0", "torchvision==0.26.0"]
DEPS = ["diffusers==0.30.0", "transformers==4.46.0", "huggingface-hub==0.30.2", "accelerate==1.1.1",
        "opencv-python-headless", "omegaconf", "einops", "trimesh", "pymeshlab", "pygltflib", "xatlas",
        "pytorch-lightning", "scikit-image", "safetensors", "scipy", "spandrel", "fast_simplification",
        "pybind11", "numpy", "pillow", "tqdm"]
ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def sh(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, env=env)


def uv():
    found = shutil.which("uv")
    if found:
        return found
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    return shutil.which("uv") or str(Path(sys.executable).parent / "uv")


def fetch_prebuilt(dest):
    """Copy hy21/sm120/cp313/* from this repo's ``wheels`` branch into ``dest``; returns dest (or None)."""
    git = ["git", "-C", ROOT]
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        git += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}"]
    folder = "hy21/sm120/cp313"
    if subprocess.run([*git, "fetch", "-q", "--depth", "1", "origin", "wheels"]).returncode != 0:
        return None
    names = subprocess.run([*git, "ls-tree", "--name-only", "FETCH_HEAD", f"{folder}/"],
                           capture_output=True, text=True).stdout.split()
    if not names:
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        data = subprocess.run([*git, "show", f"FETCH_HEAD:{name}"], capture_output=True, check=True).stdout
        (dest / Path(name).name).write_bytes(data)
    print(f"prebuilt hy21 extensions: {[Path(n).name for n in names]}", flush=True)
    return dest


def setup(base, clone):
    """Install the hy21 venv, repo, extensions and upscaler checkpoint under ``base``; returns the venv python."""
    base = Path(base)
    venv = base / "hy21-venv"
    py = venv / "bin" / "python"
    src = base / "Hunyuan3D-2.1"
    done = base / f".hy21-ready-{COMMIT[:12]}"
    if done.exists() and py.exists():
        return py, src

    if not py.exists():
        sh([uv(), "venv", "-q", "--python", PYTHON, venv])
    sh([uv(), "pip", "install", "-q", "--python", py, *TORCH, *DEPS])
    clone(REPO, COMMIT, src)
    paint = src / "hy3dpaint"

    prebuilt = fetch_prebuilt(base / "hy21-prebuilt")
    whl = sorted(prebuilt.glob("custom_rasterizer-*.whl")) if prebuilt else []
    if whl:
        sh([uv(), "pip", "install", "-q", "--python", py, "--no-deps", whl[-1]])
    else:  # compile with the venv's torch; needs nvcc on PATH
        sh([uv(), "pip", "install", "-q", "--python", py, "--no-build-isolation", paint / "custom_rasterizer"])

    renderer = paint / "DifferentiableRenderer"
    so = sorted(prebuilt.glob("mesh_inpaint_processor*.so")) if prebuilt else []
    if so:
        shutil.copy(so[-1], renderer / so[-1].name)
    elif not list(renderer.glob("mesh_inpaint_processor*.so")):
        inc = subprocess.run([py, "-m", "pybind11", "--includes"], capture_output=True, text=True, check=True).stdout.split()
        suffix = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
                                capture_output=True, text=True, check=True).stdout.strip()
        sh(["c++", "-O3", "-shared", "-std=c++14", "-fPIC", *inc, "mesh_inpaint_processor.cpp",
            "-o", f"mesh_inpaint_processor{suffix}"], cwd=renderer)

    ckpt = paint / "ckpt" / "RealESRGAN_x4plus.pth"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ESRGAN}", flush=True)
        urllib.request.urlretrieve(ESRGAN, ckpt.with_suffix(".part"))
        ckpt.with_suffix(".part").rename(ckpt)
    sh([py, "-c", "import torch, custom_rasterizer, diffusers; print('hy21 venv ok, torch', torch.__version__)"])
    done.touch()
    return py, src


def paint(py, src, mesh_path, images, out_path, o, stage, work):
    """Run hy21_runner.py in the hy21 venv; ``images`` are RGBA cutouts, front first."""
    stage("painting with Hunyuan3D-2.1 (PBR)")
    cmd = [py, ROOT / "hy21_runner.py", "--repo", src, "--mesh", mesh_path, "--out", out_path, "--work", work,
           "--faces", int(o.get("hy21_faces") or 150_000), "--views", int(o.get("hy21_views") or 8)]
    for im in images:
        cmd += ["--image", im]
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    from trellis_engine import sh_tail
    sh_tail(cmd, env=env)

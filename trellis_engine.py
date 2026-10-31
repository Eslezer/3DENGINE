"""TRELLIS.2 engine for worker.py: image -> textured (PBR) GLB in one pass.

Targets big recent GPUs (e.g. molab's RTX PRO 6000 Blackwell, 96 GB). The stack is
PyTorch 2.11 (CUDA 13, from PyPI) plus TRELLIS.2's CUDA extensions. Those extensions
are prebuilt for sm_120 on the ``wheels`` branch of this repo (see tools/build_trellis_wheels.sh);
on any other GPU or Python version setup compiles them from source with a pip-installed nvcc.
Attention runs on torch SDPA through the stand-in in trellis_compat/, so neither
flash-attn nor xformers is needed.

DINOv3, which TRELLIS.2 uses to read the input image, is gated on Hugging Face:
accept its license and provide HF_TOKEN.
"""

import gc
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from hy3d_queue import has_transparency

ROOT = Path(__file__).resolve().parent
TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.2"
TRELLIS_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
MODEL = "microsoft/TRELLIS.2-4B"
TORCH = ["torch==2.11.0", "torchvision==0.26.0"]
CUDA_BUILD = ["nvidia-cuda-nvcc==13.0.*", "nvidia-nvvm==13.0.*", "nvidia-cuda-crt==13.0.*",
              "nvidia-cuda-cccl==13.0.*", "ninja", "setuptools", "wheel"]
DEPS = ["imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless", "trimesh", "transformers>=4.56,<5",
        "pandas", "zstandard", "kornia", "timm", "plyfile", "numpy", "pillow", "huggingface_hub", "pygltflib", "scipy",
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"]
# name -> (git url, ref, clone recursively); refs match the prebuilt wheels
EXTENSIONS = {
    "nvdiffrast": ("https://github.com/NVlabs/nvdiffrast.git", "253ac4fcea7de5f396371124af597e6cc957bfae", False),
    "nvdiffrec": ("https://github.com/JeffreyXiang/nvdiffrec.git", "b296927cc7fd01c2ac1087c8065c4d7248f72da4", False),
    "FlexGEMM": ("https://github.com/JeffreyXiang/FlexGEMM.git", "6dd94a859c26ee8246888502eada3dd8ad85532e", True),
    "CuMesh": ("https://github.com/JeffreyXiang/CuMesh.git", "12289e1062f0603f2f0d0771b02e1395d247f26f", True),
    "o-voxel": (None, None, False),  # ships inside the TRELLIS.2 repo
}
EXT_MODULES = {"nvdiffrast": "nvdiffrast.torch", "nvdiffrec": "nvdiffrec_render.renderutils", "FlexGEMM": "flex_gemm",
               "CuMesh": "cumesh", "o-voxel": "o_voxel"}
RESOLUTIONS = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}
# Pixal3D (generator=pixal3d): TRELLIS.2-based, pixel-aligned to the input image; MIT.
PIXAL_REPO = "https://github.com/TencentARC/Pixal3D"
PIXAL_COMMIT = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"
MOGE = "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
NATTEN = "natten==0.21.0"


def sh(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, env=env)


def pip(*args):
    """pip into this interpreter; falls back to uv for pip-less (uv-managed) environments."""
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        sh([sys.executable, "-m", "pip", "install", "-q", *args])
    elif shutil.which("uv"):
        sh(["uv", "pip", "install", "-q", "--python", sys.executable, *args])
    else:
        sh([sys.executable, "-m", "ensurepip"])
        sh([sys.executable, "-m", "pip", "install", "-q", *args])


def python_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def driver_cuda():
    """Highest CUDA version the installed driver supports, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def gpu_arch():
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def clone(url, ref, dest, recursive=False):
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    sh(["git", "init", "-q"], cwd=dest)
    sh(["git", "remote", "add", "origin", url], cwd=dest)
    sh(["git", "fetch", "-q", "--depth", "1", "origin", ref], cwd=dest)
    sh(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
    if recursive:
        sh(["git", "submodule", "update", "-q", "--init", "--recursive", "--depth", "1"], cwd=dest)
    return dest


def setup(base, repo_url, auth_config=()):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    cuda = driver_cuda()
    print(f"GPU compute capability {gpu_arch()}, driver supports CUDA {cuda}", flush=True)
    if cuda and cuda < (13, 0):
        raise SystemExit(f"this GPU driver only supports CUDA {cuda}; the TRELLIS.2 stack here needs CUDA 13 "
                         "(driver 580 or newer)")

    pip(*TORCH, *DEPS)
    src = clone(TRELLIS_REPO, TRELLIS_COMMIT, base / "TRELLIS.2")

    missing = [n for n, mod in EXT_MODULES.items() if not importable(mod)]
    if missing:
        wheels = prebuilt_wheels(base, repo_url, auth_config)
        if wheels:
            pip("--no-deps", *map(str, wheels))
            missing = [n for n, mod in EXT_MODULES.items() if not importable(mod)]
    if missing:
        build_extensions(base, src, missing)
    still = [n for n, mod in EXT_MODULES.items() if not importable(mod)]
    if still:
        raise SystemExit(f"TRELLIS.2 extensions failed to install: {still}")
    print("TRELLIS.2 ready", flush=True)
    try:  # optional painter; TRELLIS.2 works without it
        import ultratex_engine
        ultratex_engine.setup(base, pip, clone)
        print("UltraTex (ComfyUI) installed; models download on the first painter=ultratex job", flush=True)
    except Exception as e:
        print(f"UltraTex setup failed, painter=ultratex will be unavailable: {e}", flush=True)
    return src


def setup_pixal(base):
    """Clone Pixal3D and add its extra packages to this (TRELLIS.2) environment; returns the checkout."""
    base = Path(base)
    src = clone(PIXAL_REPO, PIXAL_COMMIT, base / "Pixal3D")
    done = base / f".pixal-ready-{PIXAL_COMMIT[:12]}"
    if done.exists():
        return src
    pip("--no-deps", MOGE)  # only used to estimate the camera when no FOV is given
    pip("setuptools", "wheel", "cmake")
    # Pure-Python natten (no libnatten): hide the GPU so its setup.py skips the hours-long CUDA build;
    # pixal_runner routes NAF's attention to natten's flex backend instead.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NATTEN_CUDA_ARCH="")
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", NATTEN], env=env)
    else:
        sh(["uv", "pip", "install", "-q", "--python", sys.executable, "--no-build-isolation", NATTEN], env=env)
    sh([sys.executable, "-c", "import natten; print('natten', natten.__version__)"])
    done.touch()
    return src


def importable(module):
    return subprocess.run([sys.executable, "-c", f"import torch, {module}"], capture_output=True).returncode == 0


def prebuilt_wheels(base, repo_url, auth_config):
    """Wheels from this repo's ``wheels`` branch matching this Python and GPU, if any."""
    arch = (gpu_arch() or "").replace(".", "")
    dest = base / "wheels"
    if not dest.exists():
        res = subprocess.run(["git", *auth_config, "clone", "-q", "--depth", "1", "-b", "wheels", repo_url, str(dest)])
        if res.returncode != 0:
            print("no prebuilt wheels branch; will compile", flush=True)
            return []
    folder = dest / f"sm{arch}" / python_tag()
    found = sorted(folder.glob("*.whl")) if folder.exists() else []
    print(f"prebuilt wheels for sm{arch}/{python_tag()}: {[w.name for w in found] or 'none'}", flush=True)
    return found


def build_extensions(base, trellis_src, names):
    """Compile TRELLIS.2's CUDA extensions for this GPU with a pip-installed CUDA 13 toolchain."""
    print(f"compiling {names} from source; this can take 30+ minutes", flush=True)
    pip(*CUDA_BUILD)
    import site
    cuda_home = Path(site.getsitepackages()[0]) / "nvidia" / "cu13"
    # pip's CUDA ships lib/libcudart.so.13 only; torch links with -L$CUDA_HOME/lib64 -lcudart
    if not (cuda_home / "lib" / "libcudart.so").exists():
        (cuda_home / "lib" / "libcudart.so").symlink_to("libcudart.so.13")
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to("lib")
    # Link-time stubs: nvdiffrec links -lcuda (the driver, only present at runtime) and -lnvrtc.
    stubs = cuda_home / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "stub.c").write_text("void __cuda_stub(void){}\n")
    sh(["gcc", "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1", "-o", stubs / "libcuda.so", stubs / "stub.c"])
    if not (cuda_home / "lib" / "libnvrtc.so").exists():
        (cuda_home / "lib" / "libnvrtc.so").symlink_to("libnvrtc.so.13")
    arch = gpu_arch() or "12.0"
    env = dict(os.environ, CUDA_HOME=str(cuda_home), PATH=f"{cuda_home}/bin:{os.environ['PATH']}",
               TORCH_CUDA_ARCH_LIST=arch, GPU_ARCHS=f"sm_{arch.replace('.', '')}",
               MAX_JOBS=str(os.cpu_count() or 4),
               LIBRARY_PATH=f"{stubs}:{cuda_home}/lib:{os.environ.get('LIBRARY_PATH', '')}")
    for name in names:
        url, ref, recursive = EXTENSIONS[name]
        if url:
            path = clone(url, ref, base / "ext" / name, recursive)
        else:
            path = base / "ext" / name
            if not path.exists():
                shutil.copytree(trellis_src / name, path)
                shutil.rmtree(path / "third_party" / "eigen", ignore_errors=True)
                clone("https://gitlab.com/libeigen/eigen.git", "0d120561bd570925cda6aa1f22e6cae8493cc248",
                      path / "third_party" / "eigen")
        sh([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(path)], env=env)


def use_open_rembg(birefnet_cls):
    """Swap the gated, non-commercial briaai/RMBG-2.0 for the MIT BiRefNet it is built on."""
    if getattr(birefnet_cls, "_hy3d_open", False):
        return
    orig = birefnet_cls.__init__

    def __init__(self, model_name="ZhengPeng7/BiRefNet", *args, **kw):
        if model_name == "briaai/RMBG-2.0":
            model_name = "ZhengPeng7/BiRefNet"
        orig(self, model_name, *args, **kw)

    birefnet_cls.__init__ = __init__
    birefnet_cls._hy3d_open = True


def face_front(mesh, front_rgba, min_margin=0.03):
    """Turn ``mesh`` 180 degrees about Y if the side matching the input image faces -Z; returns True if turned.

    TRELLIS.2 does not guarantee that the side shown in the input image ends up facing +Z
    (the front UltraTex and the projection tools assume); it sometimes builds the object
    backwards. The geometric creases it sculpts from the image (seams, cut-outs, facial
    features) line up with the image's line art on the correct side; see tools/orient_check.py.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from orient_check import side_scores
    except ImportError:  # tools/ needs pygltflib, which workers set up before this check lack
        pip("pygltflib")
        from orient_check import side_scores

    s = side_scores(np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), np.asarray(mesh.faces),
                    front_rgba[..., :3], mask=front_rgba[..., 3] > 128)
    print(f"orientation check: line-art match seen from +Z {s[1]:.3f}, from -Z {s[-1]:.3f}", flush=True)
    if s[-1] - s[1] > min_margin:
        mesh.apply_transform(np.diag([-1.0, 1.0, -1.0, 1.0]))  # 180 degrees about Y
        return True
    return False


class TrellisEngine:
    """Lazily loads the TRELLIS.2 image-to-3D pipeline."""

    name = "trellis2"

    def __init__(self, src, base=None):
        os.environ.setdefault("ATTN_BACKEND", "xformers")  # served by trellis_compat/xformers (torch SDPA)
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for p in (str(src), str(ROOT / "trellis_compat")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        self.torch = torch
        self.pipe = None
        self.base = Path(base) if base else Path(src).parent
        self.painter = None

    def gpu(self):
        t = self.torch
        return ", ".join(t.cuda.get_device_name(i) for i in range(t.cuda.device_count())) or "cpu"

    def can_texture(self):
        return True

    def free(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def pipeline(self):
        if self.pipe is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg
            use_open_rembg(rembg.BiRefNet)
            self.pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL)
            # Keep every stage resident when the card is big; low_vram shuffles models to CPU per stage.
            vram = self.torch.cuda.get_device_properties(0).total_memory / 2**30
            self.pipe.low_vram = vram < 40
            self.pipe.cuda()
        return self.pipe

    def run(self, image_path, o, out_path, stage):
        import o_voxel
        from PIL import Image

        resolution = int(o.get("resolution") or 1024)
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {list(RESOLUTIONS)}")
        max_faces = o.get("max_faces")
        max_faces = 500_000 if max_faces is None else (max_faces or 16_000_000)
        texture_size = int(o.get("texture_size") or 4096)

        generator = o.get("generator") or "trellis2"
        image = Image.open(image_path)
        if generator == "pixal3d":
            if resolution == 512:
                raise ValueError("pixal3d runs at resolution 1024 or 1536")
            glb, front_rgba = self.run_pixal(image_path, o, stage, resolution, max_faces, texture_size)
            mesh = None
        elif generator == "trellis2":
            stage("loading TRELLIS.2" if self.pipe is None else "preparing")
            pipe = self.pipeline()
            stage(f"generating shape + PBR texture ({resolution})")
            if not has_transparency(image):
                image = image.convert("RGB")  # so TRELLIS.2 runs its background remover instead of trusting alpha
            mesh = pipe.run(image, seed=o.get("seed", 42),
                            pipeline_type=RESOLUTIONS[resolution])[0]
            mesh.simplify(16777216)  # nvdiffrast limit

            stage(f"baking {texture_size}px texture, {max_faces} faces max")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=max_faces, texture_size=texture_size,
                remesh=True, remesh_band=1, remesh_project=0, verbose=False,
            )
        else:
            raise ValueError(f"unknown generator {generator!r}; use trellis2 or pixal3d")
        if generator == "trellis2":
            front_rgba = self.cutout_array(image)
        stage("checking orientation")
        flipped = face_front(glb, front_rgba)
        info = {"vertices": int(len(glb.vertices)), "faces": int(len(glb.faces)), "textured": True,
                "turned_around": flipped,
                "engine": self.name, "generator": generator, "resolution": resolution,
                "texture_size": texture_size}
        painter = o.get("painter") or "trellis"
        if painter == "ultratex":
            info["painter"] = "ultratex"
            info["files"] = self.paint_ultratex(glb, image_path, out_path, o, stage)
        elif painter == "hy21":
            info["painter"] = "hy21"
            info["files"] = self.paint_hy21(glb, image_path, out_path, o, stage)
        elif painter == "trellis":
            stage("exporting")
            glb.export(out_path, extension_webp=True)
        else:
            raise ValueError(f"unknown painter {painter!r}; use trellis, ultratex or hy21")
        del mesh, glb
        self.free()
        return info

    def run_pixal(self, image_path, o, stage, resolution, max_faces, texture_size):
        """Pixal3D shape + texture in a subprocess (pixal_runner.py); returns (trimesh mesh, front RGBA array)."""
        import tempfile
        import numpy as np
        import trimesh
        from PIL import Image

        stage("setting up Pixal3D")
        src = setup_pixal(self.base)
        work = Path(tempfile.mkdtemp(prefix="pixal_"))
        back = o.get("_back_image")
        self.unload()  # give Pixal3D the whole GPU
        stage(f"generating with Pixal3D ({resolution}, {'front + back' if back else 'front only'})")
        raw = work / "pixal.glb"
        cmd = [sys.executable, ROOT / "pixal_runner.py", "--repo", src, "--front", image_path, "--out", raw,
               "--resolution", resolution, "--seed", o.get("seed", 42), "--faces", max_faces,
               "--texture", texture_size, "--cutouts", work]
        if back:
            cmd += ["--back", back]
        if o.get("pixal_fov") is not None:
            cmd += ["--fov", o["pixal_fov"]]
        sh(cmd, env=dict(os.environ, PYTHONPATH=""))
        front = np.asarray(Image.open(work / "cutout_0.png").convert("RGBA"))
        return trimesh.load(raw, force="mesh", process=False), front

    def unload(self):
        """Drop the TRELLIS.2 pipeline from the GPU (it reloads on the next TRELLIS.2 job)."""
        if self.pipe is not None:
            self.pipe = None
            self.free()

    def cutout_array(self, image):
        """RGBA numpy array of a PIL image with the background removed."""
        import numpy as np
        if not has_transparency(image):
            rembg = self.pipeline().rembg_model
            rembg.to("cuda")
            image = rembg(image.convert("RGB"))
        return np.asarray(image.convert("RGBA"))

    def cutout(self, image_path, dst, stage):
        """Save ``image_path`` as RGBA with the background removed (unless it is already cut out)."""
        from PIL import Image

        image = Image.open(image_path)
        if not has_transparency(image):
            stage("removing background")
            rembg = self.pipeline().rembg_model
            rembg.to("cuda")
            image = rembg(image.convert("RGB"))
        image = image.convert("RGBA")
        # Faint alpha far from the object skews UltraTex's reference alignment; drop it.
        image.putalpha(image.getchannel("A").point(lambda v: 0 if v < 32 else v))
        image.save(dst)
        return dst

    def paint_hy21(self, glb, image_path, out_path, o, stage):
        """Repaint TRELLIS.2's mesh with Hunyuan3D-2.1's PBR model; returns {name: path} debug images."""
        import tempfile
        import hy21_engine

        stage("setting up Hunyuan3D-2.1 painter")
        py, src = hy21_engine.setup(self.base, clone)
        work = Path(tempfile.mkdtemp(prefix="hy21_"))
        mesh_path = work / "mesh.glb"
        glb.export(mesh_path)
        images = [self.cutout(image_path, work / "reference.png", stage)]
        if o.get("_back_image"):
            images.append(self.cutout(o["_back_image"], work / "back.png", stage))
        self.free()
        hy21_engine.paint(py, src, mesh_path, images, out_path, o, stage, work / "run")
        files = {"result_debug_reference.png": images[0]}
        if len(images) > 1:
            files["result_debug_back_reference.png"] = images[1]
        return files

    def paint_ultratex(self, glb, image_path, out_path, o, stage):
        """Repaint TRELLIS.2's mesh with UltraTex; returns {name: path} debug images to publish."""
        import tempfile
        import ultratex_engine
        from PIL import Image

        if self.painter is None:
            if not (self.base / "ComfyUI" / "custom_nodes" / "ComfyUI-UltraTex").exists():
                stage("installing ComfyUI + UltraTex")  # workers set up before the painter existed
                ultratex_engine.setup(self.base, pip, clone)
            self.painter = ultratex_engine.UltraTexPainter(self.base)
        work = Path(tempfile.mkdtemp(prefix="ultratex_"))
        mesh_path = work / "mesh.glb"
        glb.export(mesh_path)  # PNG texture; UltraTex keeps the UVs (uv_mode keep_existing)
        ref = self.cutout(image_path, work / "reference.png", stage)
        back = self.cutout(o["_back_image"], work / "back.png", stage) if o.get("_back_image") else None
        self.free()
        debug = self.painter.paint(mesh_path, ref, out_path, Path(out_path).parent.name or "job", o, stage, back)
        files = {"result_debug_reference.png": ref}
        if back:
            files["result_debug_back_reference.png"] = back
        for i, p in enumerate(debug):
            im = Image.open(p)
            im.thumbnail((2048, 2048))
            dst = work / f"result_debug_{i}.png"
            im.save(dst)
            files[dst.name] = dst
        return files

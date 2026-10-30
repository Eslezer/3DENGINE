"""UltraTex texturing (FLUX.2-Klein base 9B + UltraTex LoRA) through a headless ComfyUI.

Used by trellis_engine when a job asks for ``painter=ultratex``: TRELLIS.2 makes the
shape, UltraTex paints six 2048 px views guided by the source image and bakes them into a
4K texture (plus metallic-roughness). See https://github.com/visualbruno/ComfyUI-UltraTex.

Licensing: FLUX.2-Klein base 9B and the UltraTex LoRAs are under the FLUX Non-Commercial
License; the Hugging Face account behind HF_TOKEN must accept it at
https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B .
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
COMFY_COMMIT = "b5a432157de6b8884f81104cd02926adf4d597e4"
ULTRATEX_REPO = "https://github.com/visualbruno/ComfyUI-UltraTex"
ULTRATEX_COMMIT = "1f2392308e7d08374cb1c3844afe7e3ee2fadbe0"
SKIP_REQS = re.compile(r"^(torch|torchvision|torchaudio|transformers)\b", re.I)  # keep trellis2's pins

FLUX_REPOS = ["Comfy-Org/Flux-2-Klein-9B", "black-forest-labs/FLUX.2-klein-base-9B"]
ULTRATEX_FILES = {  # ModelScope path -> (ComfyUI models subfolder, local name)
    "flux2/lora/dit_lora.safetensors": ("loras", "ultratex_flux2.safetensors"),
    "flux2_mr/lora/dit_lora.safetensors": ("loras", "ultratex_flux2_mr.safetensors"),
    "flux2/decoder.pt": ("ultratex", "ultratex_flux2_decoder.pt"),
}
ULTRATEX_SOURCES = [
    "https://modelscope.cn/models/Yibo-Zhang/UltraTex/resolve/master/{path}",
    "https://www.modelscope.ai/models/Yibo-Zhang/UltraTex/resolve/master/{path}",
]
ULTRATEX_HF_MIRROR = "rekuen/Ultratex-Flux2-Klein"
PORT = 8188


def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


def setup(base, pip, clone):
    """Install ComfyUI + ComfyUI-UltraTex (code only; models download on the first job)."""
    comfy = clone(COMFY_REPO, COMFY_COMMIT, Path(base) / "ComfyUI")
    reqs = [l.strip() for l in (comfy / "requirements.txt").read_text().splitlines()]
    pip(*[l for l in reqs if l and not l.startswith("#") and not SKIP_REQS.match(l)])
    node = clone(ULTRATEX_REPO, ULTRATEX_COMMIT, comfy / "custom_nodes" / "ComfyUI-UltraTex")
    pip("trimesh", "moderngl", "xatlas", "scipy", "modelscope")
    core = node / "ultratex_core"
    shutil.copyfile(ROOT / "ultratex_compat" / "render_nvdr.py", core / "render_nvdr.py")
    render = core / "render.py"
    sys.path.insert(0, str(ROOT / "ultratex_compat"))
    from render_nvdr import HOOK
    if "hy3d:" not in render.read_text():
        render.write_text(render.read_text() + HOOK)
    return comfy


# -- models ------------------------------------------------------------------------

def _pick(files, folder, pattern, prefer_fp8=True):
    """Best match for ``pattern`` among repo files under ``folder`` (or the repo root)."""
    cands = [f for f in files if re.search(pattern, f.split("/")[-1]) and (f.startswith(folder) or "/" not in f)]
    if not cands:
        return None
    # fp8 halves RAM/VRAM with little quality loss; molab notebooks get 32 GB of RAM.
    cands.sort(key=lambda f: (("fp8" in f) != prefer_fp8, len(f)))
    return cands[0]


def ensure_models(comfy, stage):
    """Download FLUX.2-Klein base 9B, its text encoder and VAE, and the UltraTex weights."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    models = Path(comfy) / "models"
    names = {}
    wanted = {
        "diffusion_models": r"flux-2-klein-base-9b.*\.safetensors$",
        "text_encoders": r"qwen_3_8b.*\.safetensors$",
        "vae": r"flux2-vae\.safetensors$",
    }
    for kind, pattern in wanted.items():
        existing = sorted((models / kind).glob("*")) if (models / kind).exists() else []
        hit = [p.name for p in existing if re.search(pattern, p.name)]
        if hit:
            names[kind] = hit[0]
            continue
        for repo in FLUX_REPOS:
            try:
                files = list_repo_files(repo)
            except (GatedRepoError, RepositoryNotFoundError) as e:
                print(f"{repo}: {type(e).__name__}", flush=True)
                continue
            f = _pick(files, f"split_files/{kind}", pattern)
            if f:
                stage(f"downloading {f.split('/')[-1]} from {repo}")
                try:
                    path = hf_hub_download(repo, f)
                except GatedRepoError:
                    raise RuntimeError(f"{repo} is gated: accept the FLUX Non-Commercial License at "
                                       f"https://huggingface.co/{repo} with the account of HF_TOKEN")
                (models / kind).mkdir(parents=True, exist_ok=True)
                os.symlink(path, models / kind / Path(f).name)
                names[kind] = Path(f).name
                break
        if kind not in names:
            raise RuntimeError(f"no {kind} file matching {pattern} in {FLUX_REPOS}; accept the FLUX.2-klein-base-9B "
                               "license at https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B")

    for src, (folder, name) in ULTRATEX_FILES.items():
        dst = models / folder / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        stage(f"downloading UltraTex {src}")
        if not _download_ultratex(src, dst):
            raise RuntimeError(f"could not download UltraTex weights {src} from ModelScope or {ULTRATEX_HF_MIRROR}")
    names["lora"], names["lora_mr"], names["decoder"] = (n for _, n in ULTRATEX_FILES.values())
    return names


def _download_ultratex(src, dst):
    tmp = dst.with_suffix(dst.suffix + ".part")
    for url in ULTRATEX_SOURCES:
        try:
            with urllib.request.urlopen(url.format(path=src), timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            if tmp.stat().st_size > 1 << 20:
                tmp.rename(dst)
                return True
        except (urllib.error.URLError, OSError) as e:
            print(f"ModelScope {url.format(path=src)}: {e}", flush=True)
    try:  # unofficial Hugging Face mirror of the same files
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(ULTRATEX_HF_MIRROR)
        match = [f for f in files if f.endswith(src)]
        if match:
            os.symlink(hf_hub_download(ULTRATEX_HF_MIRROR, match[0]), dst)
            return True
    except Exception as e:
        print(f"{ULTRATEX_HF_MIRROR}: {e}", flush=True)
    return False


# -- workflow ---------------------------------------------------------------------

def workflow(names, mesh_path, ref_name, prefix, o):
    """ComfyUI API-format graph: FLUX.2 Klein + UltraTex albedo (2048) and MR (1024) -> bake."""
    seed = int(o.get("seed", 42))
    steps = int(o.get("ultratex_steps") or 25)
    res = int(o.get("ultratex_resolution") or 2048)
    tex = int(o.get("texture_size") or 4096)
    sampler = {"vae": ["dec", 0], "positive": ["pos", 0], "prep": ["prep", 0], "steps": steps, "seed": seed,
               "guidance": 4.0, "drop_background_tokens": True, "memory_factor": 0.0,
               "attention": "ultratex_sparse", "sparse_topk": 0.2, "token_chunk": 8192, "denoise": 1.0}
    graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": names["diffusion_models"], "weight_dtype": "default"}},
        "lora": {"class_type": "UltraTexLoraLoader", "inputs": {"model": ["unet", 0], "lora_name": names["lora"], "strength": 1.0}},
        "lora_mr": {"class_type": "UltraTexLoraLoader", "inputs": {"model": ["unet", 0], "lora_name": names["lora_mr"], "strength": 1.0}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": names["vae"]}},
        "dec": {"class_type": "UltraTexForegroundDecoder", "inputs": {"vae": ["vae", 0], "decoder_name": names["decoder"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": names["text_encoders"], "type": "flux2", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "mesh": {"class_type": "UltraTexLoadMesh", "inputs": {"glb_path": str(mesh_path), "only_vertices_and_faces": False}},
        "prep": {"class_type": "UltraTexPrep", "inputs": {
            "reference_image": ["ref", 0], "reference_mask": ["ref", 1], "trimesh": ["mesh", 0], "mesh_path": "",
            "render_size": 4096, "atlas_size": tex, "uv_mode": "keep_existing",
            "reference_alignment": "fit_front_silhouette", "mask_meaning": "background (LoadImage)"}},
        "albedo": {"class_type": "UltraTexSampler", "inputs": dict(sampler, model=["lora", 0], resolution=res)},
        # Pin the front view to the reference (RePaint keep mask); the other five views are painted around it.
        "pin": {"class_type": "Hy3dPinFront", "inputs": {"prep": ["prep", 0], "resolution": res, "erode_px": 6, "feather_px": 12}},
        "mr": {"class_type": "UltraTexSampler", "inputs": dict(sampler, model=["lora_mr", 0], resolution=min(res, 1024))},
        "bake": {"class_type": "UltraTexBake", "inputs": {
            "prep": ["prep", 0], "albedo_views": ["albedo", 0], "metallic_roughness_views": ["mr", 0],
            "texture_size": tex, "filename_prefix": prefix, "view_weight_power": 4.0, "best_view_mix": 0.5,
            "edge_feather_px": 6.0, "fusion": "two_band", "detail_sharpness": 32.0, "exposure_match": True,
            "hole_fill": "mesh", "views_2_weight": 0.6}},
        # Debug images: the normal-map views (checks the renderer) and the generated albedo views.
        "save_normals": {"class_type": "SaveImage", "inputs": {"images": ["prep", 1], "filename_prefix": prefix + "_normals"}},
        "save_views": {"class_type": "SaveImage", "inputs": {"images": ["albedo", 2], "filename_prefix": prefix + "_views"}},
    }
    if o.get("pin_front", True):
        graph["albedo"]["inputs"].update(init_views=["pin", 0], keep_mask=["pin", 1], denoise=1.0)
    else:
        del graph["pin"]
    return graph


# -- ComfyUI server -------------------------------------------------------------------

class Comfy:
    def __init__(self, comfy_dir):
        self.dir = Path(comfy_dir)
        self.proc = None
        self.url = f"http://127.0.0.1:{PORT}"
        self.client = str(uuid.uuid4())

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read())

    def start(self, stage):
        if self.proc and self.proc.poll() is None:
            return
        stage("starting ComfyUI")
        self._kill_stale()
        log = open(self.dir / "comfyui.log", "a")
        self.proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
                                      "--disable-auto-launch"], cwd=self.dir, stdout=log, stderr=subprocess.STDOUT)
        (self.dir / "comfyui.pid").write_text(str(self.proc.pid))
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited; see {self.dir / 'comfyui.log'}:\n{self.tail()}")
            try:
                self._get("/system_stats")
                return
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        raise RuntimeError("ComfyUI did not start within 5 minutes:\n" + self.tail())

    def _kill_stale(self):
        """A worker that re-exec'd itself after a code update leaves its ComfyUI running (port + VRAM)."""
        pidfile = self.dir / "comfyui.pid"
        if not pidfile.exists():
            return
        try:
            pid = int(pidfile.read_text())
            os.kill(pid, 15)
            for _ in range(30):
                os.kill(pid, 0)  # raises once the process is gone
                time.sleep(1)
            os.kill(pid, 9)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        pidfile.unlink(missing_ok=True)

    def tail(self, n=40):
        p = self.dir / "comfyui.log"
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""

    def run(self, graph, stage, timeout=3600):
        body = json.dumps({"prompt": graph, "client_id": self.client}).encode()
        req = urllib.request.Request(self.url + "/prompt", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                pid = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode(errors='replace')[:3000]}")
        start, seen = time.time(), 0
        while time.time() - start < timeout:
            hist = self._get(f"/history/{pid}")
            if pid in hist:
                h = hist[pid]
                status = h.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[-1][1] if msgs else status
                    raise RuntimeError(f"UltraTex workflow failed: {json.dumps(detail)[:3000]}\n{self.tail()}")
                return h["outputs"]
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI crashed:\n" + self.tail())
            elapsed = int(time.time() - start)
            if elapsed - seen >= 60:
                seen = elapsed
                stage(f"UltraTex painting ({elapsed // 60} min)")
            time.sleep(3)
        raise RuntimeError(f"UltraTex timed out after {timeout}s")


def install_hy3d_nodes(comfy_dir):
    """(Re)install this repo's ComfyUI nodes; cheap, so done on every painter start."""
    dst = Path(comfy_dir) / "custom_nodes" / "hy3d_nodes"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ROOT / "ultratex_compat" / "hy3d_nodes", dst)


class UltraTexPainter:
    def __init__(self, base):
        self.comfy_dir = Path(base) / "ComfyUI"
        install_hy3d_nodes(self.comfy_dir)
        self.server = Comfy(self.comfy_dir)
        self.names = None

    def paint(self, mesh_glb, ref_rgba, out_glb, job_tag, o, stage):
        """Texture ``mesh_glb`` from the RGBA reference; writes the GLB and returns debug image paths."""
        if self.names is None:
            self.names = ensure_models(self.comfy_dir, stage)
        self.server.start(stage)
        inp = self.comfy_dir / "input"
        inp.mkdir(exist_ok=True)
        ref_name = f"hy3d_{job_tag}.png"
        shutil.copyfile(ref_rgba, inp / ref_name)
        prefix = f"hy3d/{job_tag}"
        stage("UltraTex painting (FLUX.2 views + bake)")
        outputs = self.server.run(workflow(self.names, mesh_glb, ref_name, prefix, o), stage)
        out_dir = self.comfy_dir / "output"
        glbs = [out_dir / i["subfolder"] / i["filename"] for i in outputs.get("bake", {}).get("3d", [])]
        if not glbs:
            glbs = sorted(out_dir.glob(f"{prefix}_*.glb"))
        if not glbs:
            raise RuntimeError(f"UltraTex produced no GLB; outputs: {json.dumps(outputs)[:2000]}")
        shutil.copyfile(glbs[-1], out_glb)
        debug = []
        for key in ("save_normals", "save_views"):
            for i in outputs.get(key, {}).get("images", []):
                debug.append(out_dir / i["subfolder"] / i["filename"])
        return debug

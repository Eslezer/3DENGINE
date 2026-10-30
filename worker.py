#!/usr/bin/env python3
"""3D worker: polls the ``queue`` branch of this repo and turns images into GLBs.

Two engines:
  hunyuan  Hunyuan3D-2 shape + paint; fits 16 GB cards (Kaggle, see kaggle/hy3d_worker.ipynb)
  trellis2 TRELLIS.2 shape + PBR texture; wants 24 GB+ and CUDA 13 (molab, see molab/trellis_worker.py)

    GITHUB_TOKEN=... python3 worker.py --setup --texture            # Hunyuan, with texturing
    GITHUB_TOKEN=... HF_TOKEN=... python3 worker.py --setup --engine trellis2
    GITHUB_TOKEN=... python3 worker.py [--engine trellis2]          # serve jobs forever

Assumes a single worker: on startup, jobs left "claimed"/"running" by a previous
session are re-queued.
"""

import argparse
import base64
import gc
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from hy3d_queue import Queue, now

ROOT = Path(__file__).resolve().parent
HY3D_REPO = "https://github.com/Tencent/Hunyuan3D-2"

# preset -> (HF repo, subfolder, default steps, flashvdm)
PRESETS = {
    "v2": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", 30, False),
    "v2-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo", 5, True),
    "mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini", 30, False),
    "mini-turbo": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo", 5, True),
    "v2-1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", 30, False),
}
PAINT_VARIANTS = {"full": "hunyuan3d-paint-v2-0", "turbo": "hunyuan3d-paint-v2-0-turbo"}
SKIP_REQS = re.compile(r"^(torch|torchvision|torchaudio|gradio|fastapi|uvicorn|bpy)\b", re.I)


def sh(cmd, cwd=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def setup(base, texture):
    Path(base).mkdir(parents=True, exist_ok=True)
    src = Path(base) / "Hunyuan3D-2"
    if not src.exists():
        sh(f"git clone --depth 1 {HY3D_REPO} {src}")
    reqs = [l.strip() for l in (src / "requirements.txt").read_text().splitlines()]
    reqs = [l for l in reqs if l and not l.startswith("#") and not SKIP_REQS.match(l)]
    sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{r}'" for r in reqs))
    sh(f"{sys.executable} -m pip install -q -e .", cwd=src)
    if texture:
        for ext in ("custom_rasterizer", "differentiable_renderer"):
            sh(f"{sys.executable} setup.py install", cwd=src / "hy3dgen" / "texgen" / ext)
    return src


def allow_custom_pipelines():
    """Newer diffusers refuses Hunyuan's bundled custom paint pipeline without trust_remote_code."""
    from diffusers import DiffusionPipeline
    if getattr(DiffusionPipeline, "_hy3d_trusts_code", False):
        return
    orig = DiffusionPipeline.from_pretrained.__func__

    def from_pretrained(cls, *args, **kw):
        kw.setdefault("trust_remote_code", True)
        return orig(cls, *args, **kw)

    DiffusionPipeline.from_pretrained = classmethod(from_pretrained)
    DiffusionPipeline._hy3d_trusts_code = True


def code_updated(auth_config):
    """Fast-forward this checkout of the repo; True if new code arrived."""
    git_auth = ["git", *auth_config, "-C", str(ROOT)]
    before = subprocess.run(git_auth + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout
    res = subprocess.run(git_auth + ["pull", "-q", "--ff-only"], capture_output=True, text=True)
    if res.returncode != 0:
        print("self-update failed:", res.stderr.strip(), flush=True)
        return False
    after = subprocess.run(git_auth + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout
    return before != after


class Engine:
    """Lazily loads and caches Hunyuan3D pipelines."""

    name = "hunyuan"

    def __init__(self):
        import torch
        self.torch = torch
        self.shape = {}
        self.paint = None
        self.paint_variant = None
        self.offload = False
        self.rembg = None

    def can_texture(self):
        try:
            import custom_rasterizer  # noqa: F401  built by `worker.py --setup --texture`
            return True
        except ImportError:
            return False

    def gpu(self):
        t = self.torch
        return ", ".join(t.cuda.get_device_name(i) for i in range(t.cuda.device_count())) or "cpu"

    def shape_pipe(self, preset):
        if preset not in self.shape:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            repo, sub, _, flash = PRESETS[preset]
            self.shape.clear()  # one shape model in VRAM at a time
            self.free()
            try:
                pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo, subfolder=sub, use_safetensors=True)
            except Exception as e:  # some variants only ship .ckpt weights
                print("safetensors load failed, retrying with ckpt:", e)
                pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo, subfolder=sub, use_safetensors=False)
            if flash:
                try:
                    pipe.enable_flashvdm()
                except Exception as e:
                    print("flashvdm unavailable:", e)
            self.shape[preset] = pipe
        return self.shape[preset]

    def paint_pipe(self, variant):
        if variant not in PAINT_VARIANTS:
            raise ValueError(f"unknown paint variant {variant!r}; choose from {list(PAINT_VARIANTS)}")
        if self.paint_variant != variant:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            allow_custom_pipelines()
            self.paint = None
            self.free()
            self.paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2",
                                                                subfolder=PAINT_VARIANTS[variant])
            self.paint_variant = variant
            if self.offload:
                self.paint.enable_model_cpu_offload()
        return self.paint

    def texture(self, mesh, image, variant, stage):
        """Paint ``mesh`` with the shape models parked on the CPU so paint gets the whole GPU."""
        parked = list(self.shape.values())
        for p in parked:
            p.to("cpu")
        self.free()
        try:
            stage(f"painting texture ({variant})")
            try:
                return self.paint_pipe(variant)(mesh, image=image)
            except self.torch.cuda.OutOfMemoryError:
                if self.offload:
                    raise
                # Fall back to sequential CPU offload: slower, but fits ~16 GB cards.
                stage(f"painting texture ({variant}, out of VRAM, retrying with CPU offload)")
                self.offload = True
                self.paint_variant = None
                return self.paint_pipe(variant)(mesh, image=image)
        finally:
            self.free()
            for p in parked:
                p.to("cuda")

    def free(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def run(self, image_path, o, out_path, stage):
        from PIL import Image
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        image = Image.open(image_path)
        has_alpha = image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema()[0] < 255
        if o.get("remove_bg", True) and not has_alpha:
            stage("removing background")
            if self.rembg is None:
                from hy3dgen.rembg import BackgroundRemover
                self.rembg = BackgroundRemover()
            image = self.rembg(image.convert("RGB"))
        else:
            image = image.convert("RGBA")

        preset = o.get("model") or "v2"
        if preset not in PRESETS:
            raise ValueError(f"unknown model {preset!r}; choose from {list(PRESETS)}")
        stage(f"loading {preset}")
        pipe = self.shape_pipe(preset)
        steps = o.get("steps") or PRESETS[preset][2]
        stage(f"generating shape ({steps} steps, octree {o.get('octree_resolution', 256)})")
        mesh = pipe(
            image=image,
            num_inference_steps=steps,
            octree_resolution=o.get("octree_resolution", 256),
            guidance_scale=o.get("guidance_scale", 5.0),
            num_chunks=20000,
            generator=self.torch.manual_seed(o.get("seed", 1234)),
            output_type="trimesh",
        )[0]

        stage("cleaning mesh")
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        max_faces = 40000 if o.get("max_faces") is None else o["max_faces"]
        if max_faces:
            mesh = FaceReducer()(mesh, max_facenum=max_faces)

        if o.get("texture"):
            if not self.can_texture():
                raise RuntimeError("texture requested but the texture extensions aren't built; "
                                   "set TEXTURE = True in the notebook and rerun setup")
            mesh = self.texture(mesh, image, o.get("paint", "full"), stage)

        stage("exporting")
        mesh.export(out_path)
        self.free()
        return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                "textured": bool(o.get("texture")), "engine": self.name}


def serve(q, engine, poll, heartbeat_every, once=False, update_every=0, auth_config=()):
    session = f"{socket.gethostname()}-{os.getpid()}"

    def mine(job):
        return (job or {}).get("engine") in (None, "any", engine.name)
    gpu = engine.gpu()
    texture = engine.can_texture()

    def beat(state):
        # One heartbeat file per engine so a Kaggle and a molab worker can share the queue.
        q.publish({f"worker/heartbeat-{engine.name}.json": json.dumps(
            {"time": now(), "state": state, "gpu": gpu, "host": session, "texture": texture,
             "engine": engine.name}, indent=2)},
            f"heartbeat {engine.name} {state}", remove=["worker/heartbeat.json"])

    def set_status(job_id, **st):
        q.publish({f"jobs/{job_id}/status.json": json.dumps(
            {"time": now(), "worker": session, "engine": engine.name, **st}, indent=2)},
                  f"{job_id}: {st.get('state')} {st.get('stage', '')}".strip())

    q.sync()
    for j in q.jobs():  # re-queue work orphaned by a previous session of this engine
        st = q.status(j) or {}
        if st.get("state") in ("claimed", "running") and st.get("engine", "hunyuan") == engine.name:
            q.publish({}, f"requeue {j}", remove=[f"jobs/{j}/status.json"])
    beat("idle")
    last_beat = last_update = time.time()
    print(f"worker {session} ({engine.name}) on {gpu}; polling every {poll}s", flush=True)

    while True:
        try:
            q.sync()
        except RuntimeError as e:
            print("sync failed:", e, flush=True)
            time.sleep(poll)
            continue
        pending = [j for j in q.jobs() if q.status(j) is None and mine(q.job(j))]
        if not pending:
            if once:
                return
            if update_every and time.time() - last_update > update_every:
                last_update = time.time()
                if code_updated(auth_config):
                    print("new worker code on main; restarting", flush=True)
                    os.execv(sys.executable, [sys.executable, "-u", *sys.argv])
            if time.time() - last_beat > heartbeat_every:
                beat("idle")
                last_beat = time.time()
            time.sleep(poll)
            continue

        job_id = pending[0]
        job = q.job(job_id)
        print(f"== {job_id} {job.get('options')}", flush=True)
        set_status(job_id, state="claimed")
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / job["image"]
            image.write_bytes((q.dir / "jobs" / job_id / job["image"]).read_bytes())
            out = Path(tmp) / "result.glb"

            def stage(msg):
                print(f"   {msg}", flush=True)
                set_status(job_id, state="running", stage=msg)

            try:
                info = engine.run(image, job.get("options", {}), out, stage)
                info["seconds"] = round(time.time() - started, 1)
                size = out.stat().st_size
                if size > 95 * 1024 * 1024:
                    raise RuntimeError(f"result is {size >> 20} MB, over GitHub's 100 MB limit; "
                                       "lower max_faces or skip texture")
                q.publish({
                    f"jobs/{job_id}/result.glb": out,
                    f"jobs/{job_id}/status.json": json.dumps(
                        {"time": now(), "worker": session, "engine": engine.name, "state": "done", **info}, indent=2),
                }, f"{job_id}: done")
                print(f"   done in {info['seconds']}s: {info}", flush=True)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb, flush=True)
                q.publish({
                    f"jobs/{job_id}/log.txt": tb,
                    f"jobs/{job_id}/status.json": json.dumps(
                        {"time": now(), "worker": session, "engine": engine.name, "state": "error",
                         "error": f"{type(e).__name__}: {e}"}, indent=2),
                }, f"{job_id}: error")
                engine.free()
        beat("idle")
        last_beat = time.time()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["hunyuan", "trellis2"], default=os.environ.get("HY3D_ENGINE", "hunyuan"))
    p.add_argument("--setup", action="store_true", help="install the engine before serving")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--texture", action="store_true", help="with --setup: build texture CUDA extensions")
    # Not /kaggle/working: that directory is saved as notebook output.
    p.add_argument("--base", default=os.environ.get("HY3D_BASE", "/tmp/hy3d"))
    p.add_argument("--repo", default=os.environ.get("HY3D_REPO", "Eslezer/3DENGINE"),
                   help="owner/name of the GitHub repo holding the queue branch")
    p.add_argument("--poll", type=int, default=10)
    p.add_argument("--heartbeat", type=int, default=600, help="seconds between idle heartbeats")
    p.add_argument("--once", action="store_true", help="exit when the queue is empty")
    p.add_argument("--update-every", type=int, default=60,
                   help="seconds between checks for new code on main while idle (0 = never)")
    a = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("set GITHUB_TOKEN (a PAT with Contents read/write on the repo)")
    # Auth via header rather than the URL so the token never shows up in git error output.
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    header = f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}"
    repo_url = f"https://github.com/{a.repo}.git"

    if a.engine == "trellis2":
        import trellis_engine
        if a.setup or a.setup_only:
            trellis_engine.setup(a.base, repo_url, ["-c", header])
            if a.setup_only:
                return
        engine = trellis_engine.TrellisEngine(Path(a.base) / "TRELLIS.2")
    else:
        if a.setup or a.setup_only:
            src = setup(a.base, a.texture)
            sys.path.insert(0, str(src))
            if a.setup_only:
                return
        engine = Engine()

    q = Queue(Path(a.base) / "queue", repo_url, name="hy3d-worker", email="hy3d-worker@localhost",
              config={"http.https://github.com/.extraheader": f"AUTHORIZATION: basic {auth}"}).ensure()
    serve(q, engine, a.poll, a.heartbeat, once=a.once, update_every=a.update_every, auth_config=["-c", header])


if __name__ == "__main__":
    main()

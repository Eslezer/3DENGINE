#!/usr/bin/env python3
"""4x upscale images with Real-ESRGAN (anime 6B by default), tiled, on GPU if present else CPU.

    python3 tools/upscale.py in.png out.png [in2.png out2.png ...] [--max-side 4096]

Used by remix.py so projected drawings and textures keep crisp line art. Needs torch and
spandrel (CPU is fine: pip install --break-system-packages spandrel torch --index-url
https://download.pytorch.org/whl/cpu ; spandrel itself comes from PyPI). The checkpoint is
downloaded once to ~/.cache/hy3d/.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = {
    "anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "photo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}
_loaded = {}


def checkpoint(kind):
    url = MODELS[kind]
    dst = Path.home() / ".cache" / "hy3d" / url.rsplit("/", 1)[1]
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dst.with_suffix(".part"))
        dst.with_suffix(".part").rename(dst)
    return dst


def model(kind="anime"):
    if kind not in _loaded:
        import torch
        from spandrel import ModelLoader
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _loaded[kind] = (ModelLoader().load_from_file(str(checkpoint(kind))).to(dev).eval(), dev)
    return _loaded[kind]


def upscale(image, kind="anime", max_side=None, tile=256, pad=16):
    """PIL image -> 4x PIL image (alpha resized with Lanczos), optionally capped at ``max_side``."""
    import torch

    net, dev = model(kind)
    alpha = image.getchannel("A") if image.mode in ("RGBA", "LA") else None
    x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(dev)
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * 4, w * 4)
    with torch.no_grad():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
                yb, xb = min(y0 + tile + pad, h), min(x0 + tile + pad, w)
                o = net(x[:, :, ya:yb, xa:xb]).cpu()
                th, tw = min(tile, h - y0), min(tile, w - x0)
                out[:, :, y0 * 4:(y0 + th) * 4, x0 * 4:(x0 + tw) * 4] = \
                    o[:, :, (y0 - ya) * 4:(y0 - ya + th) * 4, (x0 - xa) * 4:(x0 - xa + tw) * 4]
    res = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))
    if alpha is not None:
        res.putalpha(alpha.resize(res.size, Image.Resampling.LANCZOS))
    if max_side and max(res.size) > max_side:
        k = max_side / max(res.size)
        res = res.resize((round(res.width * k), round(res.height * k)), Image.Resampling.LANCZOS)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pairs", nargs="+", help="in out [in out ...]")
    p.add_argument("--kind", choices=list(MODELS), default="anime")
    p.add_argument("--max-side", type=int)
    a = p.parse_args()
    if len(a.pairs) % 2:
        p.error("give input/output pairs")
    for src, dst in zip(a.pairs[::2], a.pairs[1::2]):
        upscale(Image.open(src), a.kind, a.max_side).save(dst)
        print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()

"""Headless stand-in for UltraTex's moderngl G-buffer renderer, built on nvdiffrast.

moderngl needs an OpenGL context, which GPU notebook containers (molab, Kaggle) usually
don't expose. ultratex_engine.setup() copies this file into ComfyUI-UltraTex's
ultratex_core/ and appends a hook to render.py: if moderngl cannot create a context,
GLRenderer becomes ``make(GLRenderer)``.

Same outputs as the GLSL shaders: per pixel (position, 1), (normalised normal, 1),
(view depth, u, v, 1) with coverage in alpha and zeros elsewhere, plus ``last_face``
(triangle index). Like the original, rows are flipped from OpenGL's bottom-up order.
"""

import numpy as np
import torch

_CTX = None


def _context():
    global _CTX
    if _CTX is None:
        import nvdiffrast.torch as dr
        _CTX = dr.RasterizeCudaContext()
    return _CTX


def make(base):
    class NvdrRenderer(base):
        def __init__(self, vertices, normals, faces, uvs=None):  # noqa: D401 - same signature as base
            dev = torch.device("cuda")
            if uvs is None:
                uvs = np.zeros((len(vertices), 2))
            self.pos = torch.as_tensor(np.asarray(vertices, np.float32), device=dev)
            self.nrm = torch.as_tensor(np.asarray(normals, np.float32), device=dev)
            self.uv = torch.as_tensor(np.asarray(uvs, np.float32), device=dev)
            self.tri = torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int32), device=dev)
            self.last_face = None

        def release(self):
            self.pos = self.nrm = self.uv = self.tri = None

        def _render(self, size, mvp, mv, uv_mode, read=(0, 1, 2, 3)):
            import nvdiffrast.torch as dr

            dev = self.pos.device
            ones = torch.ones_like(self.pos[:, :1])
            if uv_mode:
                clip = torch.cat([self.uv * 2 - 1, torch.zeros_like(ones), ones], 1)
            else:
                m = torch.as_tensor(np.asarray(mvp, np.float32), device=dev)
                clip = torch.cat([self.pos, ones], 1) @ m.T
            v = torch.as_tensor(np.asarray(mv, np.float32), device=dev)
            depth = -(torch.cat([self.pos, ones], 1) @ v.T)[:, 2:3]
            with torch.no_grad():
                rast, _ = dr.rasterize(_context(), clip[None].contiguous(), self.tri, (size, size))
                attrs = torch.cat([self.pos, self.nrm, depth, self.uv], 1)[None].contiguous()
                out, _ = dr.interpolate(attrs, rast, self.tri)
            out, rast = out[0], rast[0]
            cov = (rast[..., 3] > 0).float()[..., None]

            n = out[..., 3:6]
            length = n.norm(dim=-1, keepdim=True)
            n = torch.where(length > 1e-12, n / length.clamp_min(1e-12), torch.zeros_like(n))
            bufs = [torch.cat([out[..., 0:3], cov], -1), torch.cat([n, cov], -1), torch.cat([out[..., 6:9], cov], -1)]
            # GL clears to 0 and the shaders write alpha 1 only where a triangle is drawn
            bufs = [(b * cov).flip(0).cpu().numpy().astype(np.float32) if i in read else None
                    for i, b in enumerate(bufs)]
            face = (rast[..., 3] - 1).clamp_min(0) * cov[..., 0]
            self.last_face = face.flip(0).cpu().numpy().astype(np.float32) if 3 in read else None
            return bufs

    NvdrRenderer.__name__ = "GLRenderer"
    return NvdrRenderer


HOOK = '''

# --- hy3d: fall back to nvdiffrast when no OpenGL context is available (headless GPU containers)
def _hy3d_gl_ok():
    try:
        _context()
        return True
    except Exception as e:  # moderngl raises plain Exceptions when no display/EGL is available
        import logging
        logging.getLogger(__name__).warning("UltraTex: no OpenGL context (%s); rendering with nvdiffrast", e)
        return False


if not _hy3d_gl_ok():
    from .render_nvdr import make as _hy3d_make
    GLRenderer = _hy3d_make(GLRenderer)
'''

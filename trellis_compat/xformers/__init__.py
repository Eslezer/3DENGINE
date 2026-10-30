"""Minimal stand-in for xformers, covering only what TRELLIS.2 calls, built on torch SDPA.

Real xformers wheels lag behind new PyTorch/CUDA releases (and flash-attn has no
prebuilt Blackwell wheels), so the worker puts this package on sys.path and sets
ATTN_BACKEND=xformers instead of installing either.
"""

__version__ = "0.0.0+sdpa-shim"

# Prebuilt TRELLIS.2 CUDA extensions

Wheels used by `trellis_engine.setup()` on `main`, laid out as `sm<arch>/cp<python>/`.

- Built with `tools/build_trellis_wheels.sh` (on `main`) without a GPU
- PyTorch 2.11.0 from PyPI (CUDA 13.0), pip-packaged nvcc 13.0.88
- `sm120`: Blackwell RTX / RTX PRO (e.g. molab's RTX PRO 6000), plus PTX for newer GPUs
- Sources: nvdiffrast 253ac4f (v0.4.0), nvdiffrec b296927 (renderutils), FlexGEMM 6dd94a8,
  CuMesh 12289e1, o-voxel from TRELLIS.2 75fbf01

They only work with the exact PyTorch version above; `setup()` installs it first.

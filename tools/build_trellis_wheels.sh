#!/bin/bash
# Build TRELLIS.2's CUDA extensions as wheels, without a GPU, for the wheels branch.
#
#   tools/build_trellis_wheels.sh 3.12 [12.0] [out_dir]
#
# Uses PyPI's torch 2.11 (CUDA 13) and the pip-packaged CUDA 13.0 nvcc, matching what
# trellis_engine.setup() installs. Output: <out_dir>/sm<arch>/cp<py>/*.whl
set -euo pipefail
PY=${1:?python version, e.g. 3.12}
ARCH=${2:-12.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/trellis-wheels}
OUT=${3:-$WORK/out}
TAG=cp${PY/./}
SM=sm${ARCH/./}
mkdir -p "$WORK/src" "$OUT/$SM/$TAG"

eval "$(python3 - "$HERE" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import trellis_engine as t
print(f'TORCH="{" ".join(t.TORCH)}"')
print(f'CUDA_BUILD="{" ".join(t.CUDA_BUILD)}"')
print(f'TRELLIS_COMMIT={t.TRELLIS_COMMIT}')
for name, (url, ref, rec) in t.EXTENSIONS.items():
    key = name.replace("-", "_")
    print(f'URL_{key}="{url or ""}"; REF_{key}="{ref or ""}"; REC_{key}={int(rec)}')
EOF
)"

VENV=$WORK/py$TAG
[ -d "$VENV" ] || uv venv -q -p "$PY" "$VENV"
. "$VENV/bin/activate"
uv pip install -q $TORCH $CUDA_BUILD pip numpy plyfile trimesh tqdm zstandard easydict
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export CUDA_HOME=$SP/nvidia/cu13 PATH=$SP/nvidia/cu13/bin:$PATH
# pip's CUDA ships lib/libcudart.so.13 only; torch links with -L$CUDA_HOME/lib64 -lcudart
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"; ln -sfn lib "$CUDA_HOME/lib64"
export TORCH_CUDA_ARCH_LIST="$ARCH+PTX" GPU_ARCHS="$SM" MAX_JOBS=${MAX_JOBS:-$(nproc)}
export LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}

# Link-time stubs: nvdiffrec links -lcuda (the driver, only present at runtime) and -lnvrtc.
mkdir -p "$CUDA_HOME/stubs"
echo 'void __cuda_stub(void){}' > "$CUDA_HOME/stubs/stub.c"
gcc -shared -fPIC -Wl,-soname,libcuda.so.1 -o "$CUDA_HOME/stubs/libcuda.so" "$CUDA_HOME/stubs/stub.c"
ln -sf libnvrtc.so.13 "$CUDA_HOME/lib/libnvrtc.so"
export LIBRARY_PATH=$CUDA_HOME/stubs:${LIBRARY_PATH:-}
fetch() {  # url ref dest recursive
  [ -d "$3" ] && return
  git init -q "$3" && git -C "$3" remote add origin "$1"
  git -C "$3" fetch -q --depth 1 origin "$2" && git -C "$3" checkout -q FETCH_HEAD
  [ "$4" = 1 ] && git -C "$3" submodule update -q --init --recursive --depth 1 || true
}
fetch https://github.com/microsoft/TRELLIS.2 "$TRELLIS_COMMIT" "$WORK/src/TRELLIS.2" 0
if [ ! -d "$WORK/src/o-voxel" ]; then
  cp -r "$WORK/src/TRELLIS.2/o-voxel" "$WORK/src/o-voxel" && rm -rf "$WORK/src/o-voxel/third_party/eigen"
  fetch https://gitlab.com/libeigen/eigen.git 0d120561bd570925cda6aa1f22e6cae8493cc248 "$WORK/src/o-voxel/third_party/eigen" 0
fi
for name in nvdiffrast nvdiffrec FlexGEMM CuMesh o-voxel; do
  key=${name//-/_}; url=URL_$key; ref=REF_$key; rec=REC_$key
  [ -n "${!url}" ] && fetch "${!url}" "${!ref}" "$WORK/src/$name" "${!rec}"
  echo "=== $name $(date +%T)"
  rm -rf "$WORK/src/$name/build"
  pip wheel -q --no-build-isolation --no-deps -w "$OUT/$SM/$TAG" "$WORK/src/$name"
done
ls -la "$OUT/$SM/$TAG"

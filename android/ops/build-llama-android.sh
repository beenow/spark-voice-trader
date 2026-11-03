#!/bin/bash
# Cross-compile llama.cpp for Android and drop the .so files into app/src/main/jniLibs, plus the
# matching public headers into app/src/main/cpp/include (so the two cannot drift).
#
# Separate from Gradle on purpose: the libraries change rarely, and building them inside every app
# build adds minutes. `--target llama` only (the llama.cpp app/ subproject does not cross-compile).
#
# Source: third_party/llama.cpp at the pinned tag (scripts/fetch_llama_cpp.sh), or $LLAMA_CPP.
# CPU flags: armv8.2-a + dotprod + fp16 (every Snapdragon since the 855, so every Galaxy Fold).
# Without an explicit arch a cross build targets plain armv8-a and the quantized kernels lose
# their dot-product path. Override with GGML_ARM_ARCH.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK:-/opt/homebrew/share/android-commandlinetools/ndk/29.0.14206865}"
SRC="${LLAMA_CPP:-$HERE/third_party/llama.cpp}"
OUT="${OUT:-$HERE/app/src/main/jniLibs}"
ABI="${1:-arm64-v8a}"
ARCH="${GGML_ARM_ARCH:-armv8.2-a+dotprod+fp16}"
BUILD="${BUILD_DIR:-${TMPDIR:-/tmp}/leaps-llama-android-$ABI}"
CMAKE="${CMAKE:-$(command -v cmake || echo /opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake)}"

[ -d "$SRC" ] || "$HERE/scripts/fetch_llama_cpp.sh"

"$CMAKE" -B "$BUILD" -S "$SRC" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$ARCH" \
  -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384"
"$CMAKE" --build "$BUILD" --target llama -j 8

mkdir -p "$OUT/$ABI"
cp "$BUILD"/bin/*.so "$OUT/$ABI/"
echo "wrote $OUT/$ABI:"
ls -la "$OUT/$ABI"

INC="$HERE/app/src/main/cpp/include"
mkdir -p "$INC"
cp "$SRC/include/llama.h" "$INC/"
cp "$SRC"/ggml/include/*.h "$INC/"
echo "refreshed headers in $INC"

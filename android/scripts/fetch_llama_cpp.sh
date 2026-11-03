#!/bin/sh
# Clone llama.cpp at the pinned release tag into android/third_party/llama.cpp (not committed).
set -e
TAG=b10643
cd "$(dirname "$0")/.."
[ -d third_party/llama.cpp ] && { echo "third_party/llama.cpp exists"; exit 0; }
mkdir -p third_party
git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp third_party/llama.cpp

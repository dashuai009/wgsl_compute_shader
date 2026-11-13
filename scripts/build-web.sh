#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PROFILE="${PROFILE:-wasm-release}"

export PATH="${HOME}/.cargo/bin:${PATH}"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "Error: wasm-bindgen CLI not found. Install matching version with:" >&2
  echo "  cargo install wasm-bindgen-cli --version 0.2.130 --locked" >&2
  exit 1
fi

echo "==> Building wgsl_softbody for wasm32-unknown-unknown (${PROFILE})..."
cargo build \
  --manifest-path "${ROOT_DIR}/Cargo.toml" \
  --profile "${PROFILE}" \
  --target wasm32-unknown-unknown \
  --bin wgsl_softbody

WASM_PATH="${ROOT_DIR}/target/wasm32-unknown-unknown/${PROFILE}/wgsl_softbody.wasm"

echo "==> Packaging web bundle into ${DIST_DIR}..."
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/pkg"

wasm-bindgen \
  --target web \
  --no-typescript \
  --remove-name-section \
  --remove-producers-section \
  --out-dir "${DIST_DIR}/pkg" \
  --out-name wgsl_softbody \
  "${WASM_PATH}"

if command -v wasm-opt >/dev/null 2>&1; then
  echo "==> Running wasm-opt -Oz..."
  wasm-opt -Oz --all-features \
    "${DIST_DIR}/pkg/wgsl_softbody_bg.wasm" \
    -o "${DIST_DIR}/pkg/wgsl_softbody_bg.wasm"
fi

cp "${ROOT_DIR}/web/index.html" "${DIST_DIR}/index.html"
cp -R "${ROOT_DIR}/assets" "${DIST_DIR}/assets"
touch "${DIST_DIR}/.nojekyll"

echo "==> Done! Bundle size summary:"
ls -lh "${DIST_DIR}/pkg/"

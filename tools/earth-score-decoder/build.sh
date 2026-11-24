#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CC_wasm32_unknown_unknown=clang
export CFLAGS_wasm32_unknown_unknown="-O3 -msimd128 -mbulk-memory"
export RUSTFLAGS="-C target-feature=+simd128,+bulk-memory"
cargo build --locked --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/earth_score_decoder.wasm ../../src/earth-score/zstd.wasm

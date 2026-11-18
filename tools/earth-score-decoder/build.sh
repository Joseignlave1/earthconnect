#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CC_wasm32_unknown_unknown=clang
cargo build --locked --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/earth_score_decoder.wasm ../../src/earth-score/zstd.wasm

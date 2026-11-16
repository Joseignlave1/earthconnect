#!/usr/bin/env bash
# Build the pinned authors' implementation outside the project. No system install.
set -euo pipefail
upstream="${1:-/tmp/earth-index-k2-upstream}"
commit=fac218aab831f2611b5ae94f09f0ef12d1cd29ea
if [[ ! -d "$upstream/.git" ]]; then
  git clone https://gitlab.lbd.udc.es/fsilva/k2-raster "$upstream"
fi
test "$(git -C "$upstream" rev-parse HEAD)" = "$commit"
# GCC 16 diagnoses pre-existing member-name typos in SDSL and raster copy/move
# helpers. No codec/build/query algorithm changes; explicit for reproducibility.
python - "$upstream" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'external/sdsl-lite/include/sdsl/louds_tree.hpp'
s = p.read_text().replace('tree.m_select1', 'tree.m_bv_select1').replace('tree.m_select0', 'tree.m_bv_select0')
p.write_text(s)
p = pathlib.Path(sys.argv[1]) / 'include/k2_raster_base.hpp'
s = p.read_text().replace('tr.k_acum_min_values', 'tr.k_accum_min_values').replace('tr.k_acum_max_values', 'tr.k_accum_max_values')
p.write_text(s)
PY
cmake -S "$upstream" -B "$upstream/build" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$upstream/build" --target encode_k2r -j 4
here="$(cd "$(dirname "$0")" && pwd)"
g++ -std=c++11 -O3 -DNDEBUG -msse4.2 \
  -I "$upstream/include" -I "$upstream/build/external/sdsl-lite/include" \
  -I "$upstream/build/external/sdsl-lite/external/libdivsufsort/include" \
  "$here/k2_demo_query.cpp" -L "$upstream/build/lib" \
  -lsdsl -ldivsufsort -ldivsufsort64 -o "$upstream/build/bin/earth_index_demo"
printf 'Demo binary: %s/build/bin/earth_index_demo\n' "$upstream"

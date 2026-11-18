# Earth Score reference data

The independent input is the original field-major uint16 matrix from the frozen
Earth Index research export. The shipped manifest preserves the site order from
that export, which uses database region ID order and ascending pixel index.
`reference/manifest.json` pins the original source-code hashes along with the
block-packed chunk and Zstd-frame hashes.

Each `reference/chunk-NN.zst` contains two consecutive little-endian uint16
arrays of the chunk's row count. The first array is the original NumPy baseline.
The second array uses the same formula and a slope target raised by 0.25°. Both
were evaluated from the original uint16 matrix, with the exact decoded float32
measurements passed to the float64 scoring formula. Scores use round-to-even
quantization. The changed array also matches all 9,216 published changed-query
reference samples. The WebGPU comparison accepts a difference of one score
step and requires identical missingness.

The complete verification uses the included world and reference files. The
original dense matrix and database are provenance for the fixture and are
unnecessary to reproduce the comparison.

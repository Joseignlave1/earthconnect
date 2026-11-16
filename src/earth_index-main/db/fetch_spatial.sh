#!/bin/sh
set -eu
# Pinned to the same engine release as duckdb-rs and the serving file.
case "$(uname -m)" in x86_64|amd64) ;; *) echo 'Spatial package currently supports Linux amd64' >&2; exit 1 ;; esac
spatial_output=${1:-db/runtime/spatial.duckdb_extension}
mkdir -p "$(dirname "$spatial_output")"
spatial_archive="${spatial_output}.gz.part"
trap 'rm -f "$spatial_archive" "${spatial_output}.part"' EXIT
curl --retry 3 -fsSL https://extensions.duckdb.org/v2.0.0/linux_amd64/spatial.duckdb_extension.gz -o "$spatial_archive"
printf '%s  %s\n' 8d4bf1ab198806f123821d52ff5131745baaa19d5e91bd6b9772db922d31a228 "$spatial_archive" | sha256sum -c -
gzip -dc "$spatial_archive" > "${spatial_output}.part"
mv "${spatial_output}.part" "$spatial_output"

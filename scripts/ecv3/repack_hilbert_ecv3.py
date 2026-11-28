#!/usr/bin/env python3
"""Lossless 2D Hilbert unified-spatial repacker for ECV3 packs on R2.

Zero AV1 re-encoding: reads each existing `.ecv3` pack directly from R2 via the
authenticated temporary upload worker, reorders tile OBU payloads from row-major
`(kind, level, ty, tx)` to Hilbert unified-spatial order (grouping deepest albedo
and DEM tiles into contiguous 2D patches so 64 KiB aligned chunks pull both albedo
and normal for a spatial neighborhood in ~30-35% fewer requests), marks header
`reserved = 1`, verifies byte-for-byte tile payload integrity, and uploads back.

Idempotent & resumable: checks header `reserved == 1` via a 32-byte Range read first.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import struct
import sys
import time
import urllib.error
import urllib.request

ECV3_MAGIC = b"ECV3"
ECV3_VERSION = 3
ECV3_LAYOUT_HILBERT = 1

EARTH_LAND_MASK = [
    0x03FF7FC0,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFF8FF7,
    0x3FFFA7F0,
    0x3FFFC7E3,
    0xDFFFE7E7,
    0xFFDFCFC7,
    0xFF7F7F8F,
    0xFD3EBE0F,
    0xFC1E1F5D,
    0xFC46CF01,
    0xC0593600,
    0x5FB03E00,
    0xFFFFFFFF,
    0xFFFFFFFF,
]


def hilbert2d(x: int, y: int, order: int = 10) -> int:
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = (1 << order) - 1 - x
                y = (1 << order) - 1 - y
            x, y = y, x
        s >>= 1
    return d


def all_pack_keys() -> list[str]:
    keys: list[str] = []
    for i in range(48):
        keys.append(f"ecv3/moon/moon-ecv3-p{i:02d}.ecv3")
    for i in range(48):
        keys.append(f"ecv3/mars/mars-ecv3-p{i:02d}.ecv3")
    for i in range(512):
        if (EARTH_LAND_MASK[i >> 5] >> (i & 31)) & 1:
            keys.append(f"ecv3/earth/earth-ecv3-p{i:03d}.ecv3")
    return keys


def repack_ecv3_hilbert(data: bytes) -> bytes | None:
    if len(data) < 32:
        raise ValueError(f"Truncated ECV3 file ({len(data)} bytes)")
    hdr_fields = struct.unpack_from("<4sIHHHHHHHHII", data, 0)
    magic, ver, body_id, pack_id, ts, gut, bc, br, lc, tpl_cnt, tile_cnt, reserved = (
        hdr_fields
    )
    if magic != ECV3_MAGIC or ver != ECV3_VERSION:
        raise ValueError(f"Bad ECV3 header: {magic!r} v{ver}")
    if reserved == ECV3_LAYOUT_HILBERT:
        return None

    pos = 32
    for _ in range(tpl_cnt):
        _, tlen, _, _ = struct.unpack_from("<HHHH", data, pos)
        pos += 8 + tlen
        pos += (-(pos - 32)) % 4
    tpl_section = data[32:pos]
    idx_start = pos
    idx_end = idx_start + tile_cnt * 20
    if idx_end > len(data):
        raise ValueError(f"Index extends past EOF ({idx_end} > {len(data)})")

    records: list[tuple[int, int, int, int, int, int, int, bytes]] = []
    deep_by_kind: dict[int, int] = {}
    max_lvl = 0
    orig_hash = hashlib.sha256()

    for i in range(tile_cnt):
        key, off, ln, meta, aux = struct.unpack_from("<IIIII", data, idx_start + i * 20)
        kind = (key >> 24) & 0xFF
        lvl = (key >> 20) & 0xF
        ty = (key >> 10) & 0x3FF
        tx = key & 0x3FF
        obu = data[off : off + ln]
        if len(obu) != ln:
            raise ValueError(
                f"Truncated tile key={key:#x}: got {len(obu)} expected {ln}"
            )
        if lvl > deep_by_kind.get(kind, -1):
            deep_by_kind[kind] = lvl
        if lvl > max_lvl:
            max_lvl = lvl
        records.append((key, kind, lvl, tx, ty, meta, aux, obu))

    for rec in sorted(records, key=lambda r: r[0]):
        orig_hash.update(struct.pack("<IIII", rec[0], len(rec[7]), rec[5], rec[6]))
        orig_hash.update(rec[7])

    min_deep = min(deep_by_kind.values()) if deep_by_kind else 0

    def sort_key(
        r: tuple[int, int, int, int, int, int, int, bytes],
    ) -> tuple[int, int, int, int]:
        _, kind, lvl, tx, ty, _, _, _ = r
        if lvl < min_deep:
            return (lvl, hilbert2d(tx, ty), kind, 0)
        sh = max_lvl - lvl
        return (99, hilbert2d(tx << sh, ty << sh), lvl, kind)

    records.sort(key=sort_key)

    new_hdr = struct.pack(
        "<4sIHHHHHHHHII",
        ECV3_MAGIC,
        ECV3_VERSION,
        body_id,
        pack_id,
        ts,
        gut,
        bc,
        br,
        lc,
        tpl_cnt,
        tile_cnt,
        ECV3_LAYOUT_HILBERT,
    )
    out = bytearray(len(data))
    out[0:32] = new_hdr
    out[32:idx_start] = tpl_section

    cur_off = idx_end
    for i, (key, _, _, _, _, meta, aux, obu) in enumerate(records):
        ln = len(obu)
        struct.pack_into("<IIIII", out, idx_start + i * 20, key, cur_off, ln, meta, aux)
        out[cur_off : cur_off + ln] = obu
        cur_off += ln

    if cur_off != len(data):
        raise ValueError(f"Repacked size mismatch: {cur_off} != {len(data)}")

    # Strict post-repack verification: re-parse every tile entry and confirm identical payload SHA-256.
    verify_hash = hashlib.sha256()
    v_recs = []
    for i in range(tile_cnt):
        key, off, ln, meta, aux = struct.unpack_from("<IIIII", out, idx_start + i * 20)
        v_recs.append((key, ln, meta, aux, bytes(out[off : off + ln])))
    for key, ln, meta, aux, obu in sorted(v_recs, key=lambda r: r[0]):
        verify_hash.update(struct.pack("<IIII", key, ln, meta, aux))
        verify_hash.update(obu)
    if verify_hash.digest() != orig_hash.digest():
        raise ValueError("Post-repack SHA-256 verification failed!")

    return bytes(out)


def http_req(
    url: str,
    token: str,
    method: str = "GET",
    data: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
    retries: int = 5,
) -> tuple[int, bytes]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "earth-connect-ecv3-repack/1.0",
    }
    if extra_headers:
        headers.update(extra_headers)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(min(2.0 * (attempt + 1), 10.0))
    raise RuntimeError(f"{method} {url} failed after {retries} retries: {last_err}")


def process_key(key: str, base_url: str, token: str) -> tuple[str, str, int]:
    url = f"{base_url.rstrip('/')}/{key}"
    # Fast 32-byte header check so already-repacked files skip downloading MBs.
    status, head32 = http_req(
        url, token, method="GET", extra_headers={"Range": "bytes=0-31"}
    )
    if status == 404:
        return key, "missing", 0
    if len(head32) == 32:
        reserved = struct.unpack_from("<I", head32, 28)[0]
        if reserved == ECV3_LAYOUT_HILBERT:
            return key, "skipped", 0

    status, raw = http_req(url, token, method="GET")
    if status != 200:
        raise RuntimeError(f"Unexpected GET status {status} for {key}")
    repacked = repack_ecv3_hilbert(raw)
    if repacked is None:
        return key, "skipped", 0
    put_status, _ = http_req(
        url,
        token,
        method="PUT",
        data=repacked,
        extra_headers={"Content-Type": "application/octet-stream"},
    )
    if put_status != 200:
        raise RuntimeError(f"Unexpected PUT status {put_status} for {key}")
    return key, "repacked", len(repacked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--upload-url", default=os.environ.get("ECV3_UPLOAD_URL", ""))
    ap.add_argument("--upload-token", default=os.environ.get("ECV3_UPLOAD_TOKEN", ""))
    args = ap.parse_args()

    if not args.upload_url or not args.upload_token:
        print("Missing --upload-url or --upload-token", file=sys.stderr)
        return 2

    keys = [k for i, k in enumerate(all_pack_keys()) if i % args.shards == args.shard]
    t0 = time.perf_counter()
    done = skipped = repacked = failed = 0
    total_bytes = 0
    print(
        f"[shard {args.shard}/{args.shards}] starting {len(keys)} packs (workers={args.workers})",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_key, k, args.upload_url, args.upload_token): k
            for k in keys
        }
        for fut in concurrent.futures.as_completed(futs):
            k = futs[fut]
            try:
                _, state, nbytes = fut.result()
                done += 1
                if state == "repacked":
                    repacked += 1
                    total_bytes += nbytes
                elif state == "skipped":
                    skipped += 1
                if done % 10 == 0 or done == len(keys):
                    dt = time.perf_counter() - t0
                    mb = total_bytes / (1024 * 1024)
                    print(
                        f"[shard {args.shard}/{args.shards}] {done}/{len(keys)} "
                        f"(repacked={repacked}, skipped={skipped}, {mb:.1f} MB, {dt:.1f}s)",
                        flush=True,
                    )
            except Exception as e:
                failed += 1
                print(
                    f"[shard {args.shard}/{args.shards}] ERROR {k}: {e}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"[shard {args.shard}/{args.shards}] FINISHED in {time.perf_counter() - t0:.1f}s: "
        f"repacked={repacked} skipped={skipped} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

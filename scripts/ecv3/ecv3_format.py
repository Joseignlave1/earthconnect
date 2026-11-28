#!/usr/bin/env python3
"""Canonical ECV3 binary virtual-tile pack format & ECV3.2 adaptive AVIF encoder.

Binary layout (little-endian):
  Fixed header (32 bytes):
    magic:       4s  b'ECV3'
    version:     u32 = 3
    body_id:     u16 (0=earth, 1=moon, 2=mars)
    pack_id:     u16
    tile_size:   u16 (512)
    gutter:      u16 (2)
    base_cols:   u16
    base_rows:   u16
    level_count: u16
    tpl_count:   u16 (up to 4 shared AVIF ISOBMFF header templates)
    tile_count:  u32
    reserved:    u32 = 0

  Template table (tpl_count entries):
    tpl_id:      u16 (0=yuv400 8b, 1=yuv420 8b RGB, 2=yuva420 8b RGBA, 3=yuv400 10b height)
    tpl_len:     u16
    iloc_off:    u16 (offset of 4B big-endian extent_length in template)
    mdat_off:    u16 (offset of 4B big-endian mdat box_size in template)
    tpl_bytes:   tpl_len bytes (padded to 4-byte alignment)

  Tile index (tile_count x 20 bytes):
    key:         u32 = (kind << 24) | (level << 20) | (ty << 10) | tx
                       kind 0 = albedo/detail, kind 1 = 10-bit height DEM
    offset:      u32 (byte offset of raw AV1 OBU payload from start of file)
    length:      u32 (byte length of raw AV1 OBU payload)
    meta:        u32 = (tpl_id << 24) | (grain_amp_u8 << 16) | (quality_u8 << 8) | flags
    aux:         u32 = packed (float16 min_h, float16 max_h) for kind=1 height tiles, else 0

  Payloads: concatenated raw AV1 OBU payloads (zero per-tile container fat!).
"""

from __future__ import annotations

import io
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ECV3_MAGIC = b"ECV3"
ECV3_VERSION = 3
TILE_SIZE = 512
GUTTER = 2
INNER = TILE_SIZE - 2 * GUTTER  # 508

TPL_GRAY8 = 0
TPL_RGB8 = 1
TPL_RGBA8 = 2
TPL_HEIGHT10 = 3

BODY_IDS = {"earth": 0, "moon": 1, "mars": 2}


def ecv3_tile_key(kind: int, level: int, tx: int, ty: int) -> int:
    return (
        ((kind & 0xFF) << 24)
        | ((level & 0xF) << 20)
        | ((ty & 0x3FF) << 10)
        | (tx & 0x3FF)
    )


def decode_ecv3_key(key: int) -> tuple[int, int, int, int]:
    kind = (key >> 24) & 0xFF
    level = (key >> 20) & 0xF
    ty = (key >> 10) & 0x3FF
    tx = key & 0x3FF
    return kind, level, tx, ty


def pack_height_bounds(lo_m: float, hi_m: float) -> int:
    """Pack (lo, hi) meters into u32 as two IEEE-754 float16 values (scaled in decameters/meters cleanly)."""
    # Store raw float16 in meters? Wait: Mars Olympus Mons is +21,229 m (within float16 max 65,504 m!),
    # Hellas is -8,200 m, Moon is -9,100..+10,700 m, Earth is -11,000..+8,849 m!
    # float16 handles ±65,504 m cleanly (precision ~0.5–8 m at high alt, or store span+lo as float16).
    # Even better: store lo_m and span_m as float16 -> span is typically 50–4000m (precision 0.05–2m!).
    lo16 = np.float16(np.clip(lo_m, -65000.0, 65000.0))
    span16 = np.float16(np.clip(max(hi_m - lo_m, 1.0), 1.0, 65000.0))
    b = struct.pack("<ee", float(lo16), float(span16))
    return struct.unpack("<I", b)[0]


def unpack_height_bounds(aux: int) -> tuple[float, float]:
    lo, span = struct.unpack("<ee", struct.pack("<I", aux))
    return float(lo), float(lo + span)


def strip_avif_header(avif_bytes: bytes) -> tuple[bytes, bytes, int, int]:
    idx = avif_bytes.find(b"mdat")
    if idx < 4:
        raise ValueError("Invalid AVIF payload: missing mdat box")
    hdr = avif_bytes[: idx + 4]
    obu = avif_bytes[idx + 4 :]
    iloc_off = hdr.find(struct.pack(">I", len(obu)))
    mdat_off = len(hdr) - 8
    if iloc_off < 0:
        raise ValueError("Could not locate iloc extent_length in AVIF header")
    return hdr, obu, iloc_off, mdat_off


def reassemble_avif(tpl: bytes, iloc_off: int, mdat_off: int, obu: bytes) -> bytes:
    if not tpl:
        return obu
    buf = bytearray(tpl)
    ol = len(obu)
    buf[iloc_off : iloc_off + 4] = struct.pack(">I", ol)
    buf[mdat_off : mdat_off + 4] = struct.pack(">I", ol + 8)
    return bytes(buf) + obu


def to_luma(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def fast_bilateral_5x5(img: np.ndarray, sigma_r: float) -> np.ndarray:
    """Fast vectorized 5x5 edge-preserving bilateral filter (used ONLY as pruning/grain side-channel)."""
    if img.ndim == 2:
        f = img.astype(np.float32)
        h, w = f.shape
        pad = np.pad(f, 2, mode="edge")
        acc = np.zeros_like(f)
        wsum = np.zeros_like(f)
        inv_2sr2 = -0.5 / (sigma_r * sigma_r)
        inv_2ss2 = -0.5 / (1.5 * 1.5)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sw = float(np.exp((dx * dx + dy * dy) * inv_2ss2))
                nbr = pad[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
                diff = nbr - f
                wt = sw * np.exp((diff * diff) * inv_2sr2)
                acc += wt * nbr
                wsum += wt
        return np.clip(np.rint(acc / np.maximum(wsum, 1e-6)), 0, 255).astype(np.uint8)
    f = img.astype(np.float32)
    luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    h, w, _ = f.shape
    pad_f = np.pad(f, ((2, 2), (2, 2), (0, 0)), mode="edge")
    pad_l = np.pad(luma, 2, mode="edge")
    acc = np.zeros_like(f)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    inv_2sr2 = -0.5 / (sigma_r * sigma_r)
    inv_2ss2 = -0.5 / (1.5 * 1.5)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sw = float(np.exp((dx * dx + dy * dy) * inv_2ss2))
            dl = pad_l[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w] - luma
            wt = (sw * np.exp((dl * dl) * inv_2sr2))[:, :, None]
            acc += wt * pad_f[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
            wsum += wt
    return np.clip(np.rint(acc / np.maximum(wsum, 1e-6)), 0, 255).astype(np.uint8)


def fast_ssim_luma(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    al = to_luma(a_u8)
    bl = to_luma(b_u8)
    h, w = al.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    ab = (
        al[:h8, :w8]
        .reshape(h8 // 8, 8, w8 // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 64)
    )
    bb = (
        bl[:h8, :w8]
        .reshape(h8 // 8, 8, w8 // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 64)
    )
    mu_a, mu_b = ab.mean(axis=1), bb.mean(axis=1)
    va = ((ab - mu_a[:, None]) ** 2).mean(axis=1)
    vb = ((bb - mu_b[:, None]) ** 2).mean(axis=1)
    cov = ((ab - mu_a[:, None]) * (bb - mu_b[:, None])).mean(axis=1)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    return float(
        np.mean(
            ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
            / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
        )
    )


def measure_grain_amp(raw: np.ndarray, dn: np.ndarray) -> float:
    rl = to_luma(raw)
    dl = to_luma(dn)
    gx = np.abs(np.roll(dl, -1, axis=1) - np.roll(dl, 1, axis=1)) * 0.5
    gy = np.abs(np.roll(dl, -1, axis=0) - np.roll(dl, 1, axis=0)) * 0.5
    flat = (gx + gy) < 4.0
    if int(flat.sum()) < 256:
        return 0.0
    resid = rl[flat] - dl[flat]
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return 1.4826 * mad


def encode_visual_tile_ecv32(
    tile: np.ndarray,
    up_parent: np.ndarray | None,
    allow_prune: bool = True,
    speed: int = 8,
    target_ssim: float = 0.89,
) -> tuple[int, bytes, int, int] | None:
    """Encode a 512x512 visual tile (L, RGB, or RGBA) using ECV3.2 universal policy.
    Returns (tpl_id, avif_bytes, grain_amp_u8, used_q) or None if pruned."""
    if tile.ndim == 2:
        mode, tpl_id = "L", TPL_GRAY8
    elif tile.shape[2] == 3:
        mode, tpl_id = "RGB", TPL_RGB8
    else:
        mode, tpl_id = "RGBA", TPL_RGBA8

    t_in = tile[GUTTER:-GUTTER, GUTTER:-GUTTER]
    t_luma = to_luma(t_in)
    tile_std = float(t_luma.std())

    g_amp = 0.0
    q = 44  # base coarse level default (no parent)
    if up_parent is not None:
        up_in = up_parent[GUTTER:-GUTTER, GUTTER:-GUTTER]
        sigma_r = min(10.0, max(2.5, 0.35 * tile_std))
        dn = fast_bilateral_5x5(
            tile[:, :, :3] if mode == "RGBA" else tile,
            min(7.0, sigma_r) if mode != "L" else sigma_r,
        )
        g_amp = measure_grain_amp(tile[:, :, :3] if mode == "RGBA" else tile, dn)
        dn_in = dn[GUTTER:-GUTTER, GUTTER:-GUTTER]
        up_rgb = up_in[:, :, :3] if mode == "RGBA" else up_in
        diff = np.abs(dn_in.astype(np.float32) - up_rgb.astype(np.float32))
        diff_max = diff.max(axis=-1) if diff.ndim == 3 else diff
        rms = float(np.sqrt(np.mean(diff_max * diff_max)))
        p95 = float(np.percentile(diff_max, 95))

        contrast_scale = max(min(tile_std / 24.0, 1.0), 0.25)
        eff_rms = rms / contrast_scale
        eff_p95 = p95 / contrast_scale

        # Only prune if alpha (if present) also matches parent cleanly
        can_prune = allow_prune and (eff_rms < 2.4 and eff_p95 < 6.5)
        if can_prune and mode == "RGBA":
            adiff = np.abs(
                t_in[:, :, 3].astype(np.float32) - up_in[:, :, 3].astype(np.float32)
            )
            if float(adiff.max()) > 12.0:
                can_prune = False
        if can_prune:
            return None

        score = max(eff_rms / 5.5, eff_p95 / 14.0)
        q = int(round(26 + np.clip(score, 0.0, 1.0) * 24))

    def enc_once(qv: int) -> tuple[bytes, np.ndarray]:
        buf = io.BytesIO()
        Image.fromarray(tile, mode=mode).save(
            buf, format="AVIF", quality=qv, speed=speed
        )
        b = buf.getvalue()
        dec = np.asarray(Image.open(io.BytesIO(b)).convert(mode), dtype=np.uint8)
        return b, dec

    avif_b, dec = enc_once(q)
    dec_in = dec[GUTTER:-GUTTER, GUTTER:-GUTTER]

    # Target-SSIM floor: protect crisp 1px micro-craters/frost cracks while avoiding wasting bits on pure shot noise
    eff_target = 0.85 if g_amp >= 2.2 else target_ssim
    cur_ssim = fast_ssim_luma(t_in, dec_in)
    if cur_ssim < eff_target and q < 56:
        deficit = eff_target - cur_ssim
        q2 = min(58, q + int(round(8 + deficit * 80)))
        if q2 > q:
            q = q2
            avif_b, _ = enc_once(q)

    # Encode grain_amp as uint8 (scaled x10, so 0..25.5 -> 0..255)
    out_g = (g_amp * 0.85) if (g_amp >= 2.2 and q <= 42 and tile_std >= 12.0) else 0.0
    grain_u8 = int(np.clip(round(out_g * 10.0), 0, 255))
    return tpl_id, avif_b, grain_u8, q


def encode_height10_tile(
    dem_m: np.ndarray,
    up_parent_m: np.ndarray | None,
    allow_prune: bool = True,
    q: int = 74,
    speed: int = 6,
    px_m: float = 200.0,
) -> tuple[bytes, int, int] | None:
    """Encode a 512x512 float32 elevation tile (meters) as 10-bit grayscale AVIF via avifenc.
    Returns (avif_bytes, aux_u32_bounds, used_q) or None if pruned."""
    if allow_prune and up_parent_m is not None:
        # Slope-aware pruning: scale elevation residual threshold to pixel spacing (~0.85° RMS / 2.5° p95 slope)
        diff = np.abs(
            dem_m[GUTTER:-GUTTER, GUTTER:-GUTTER]
            - up_parent_m[GUTTER:-GUTTER, GUTTER:-GUTTER]
        )
        rms_thr = max(0.5, 0.015 * px_m)
        p95_thr = max(1.5, 0.045 * px_m)
        if (
            float(np.sqrt(np.mean(diff * diff))) < rms_thr
            and float(np.percentile(diff, 95)) < p95_thr
        ):
            return None

    lo = float(dem_m.min())
    hi = float(dem_m.max())
    span = max(hi - lo, 1.0)
    q16 = np.clip(np.rint(((dem_m - lo) / span) * 65535.0), 0, 65535).astype(np.uint16)

    with tempfile.TemporaryDirectory() as td:
        inp = f"{td}/h.png"
        outp = f"{td}/h.avif"
        Image.fromarray(q16).save(inp)
        subprocess.run(
            [
                "avifenc",
                "-c",
                "aom",
                "-q",
                str(q),
                "-s",
                str(speed),
                "-d",
                "10",
                "-y",
                "400",
                "-j",
                "1",
                inp,
                outp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        avif_b = Path(outp).read_bytes()

    aux = pack_height_bounds(lo, hi)
    return avif_b, aux, q


@dataclass
class Ecv3Record:
    key: int
    tpl_id: int
    obu: bytes
    grain_u8: int = 0
    quality: int = 0
    flags: int = 0
    aux: int = 0


def write_ecv3_pack(
    out_path: Path,
    body_id: int,
    pack_id: int,
    base_cols: int,
    base_rows: int,
    level_count: int,
    templates: dict[int, tuple[bytes, int, int]],
    records: list[Ecv3Record],
) -> int:
    """Write an atomic ECV3 binary pack (.tmp -> final) and return final byte size."""
    records = sorted(records, key=lambda r: r.key)
    tpl_items = sorted(templates.items(), key=lambda kv: kv[0])

    # Build template section (each padded to 4-byte boundary)
    tpl_section = bytearray()
    for tpl_id, (tpl_bytes, iloc_off, mdat_off) in tpl_items:
        tpl_section += struct.pack("<HHHH", tpl_id, len(tpl_bytes), iloc_off, mdat_off)
        tpl_section += tpl_bytes
        pad = (-len(tpl_section)) % 4
        if pad:
            tpl_section += b"\x00" * pad

    header_len = 32
    index_len = len(records) * 20
    data_offset = header_len + len(tpl_section) + index_len

    index_buf = bytearray()
    cur_off = data_offset
    for r in records:
        meta = (
            ((r.tpl_id & 0xFF) << 24)
            | ((r.grain_u8 & 0xFF) << 16)
            | ((r.quality & 0xFF) << 8)
            | (r.flags & 0xFF)
        )
        index_buf += struct.pack("<IIIII", r.key, cur_off, len(r.obu), meta, r.aux)
        cur_off += len(r.obu)

    hdr = struct.pack(
        "<4sIHHHHHHHHII",
        ECV3_MAGIC,
        ECV3_VERSION,
        body_id,
        pack_id,
        TILE_SIZE,
        GUTTER,
        base_cols,
        base_rows,
        level_count,
        len(tpl_items),
        len(records),
        0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(hdr)
        f.write(tpl_section)
        f.write(index_buf)
        for r in records:
            f.write(r.obu)
    tmp_path.replace(out_path)
    return out_path.stat().st_size


def verify_ecv3_pack(path: Path, sample_n: int = 8) -> dict:
    """Verify binary structure and decode sample tiles via template reassembly."""
    data = path.read_bytes()
    magic, ver, body_id, pack_id, ts, gut, bc, br, lc, tpl_cnt, tile_cnt, _ = (
        struct.unpack_from("<4sIHHHHHHHHII", data, 0)
    )
    assert magic == ECV3_MAGIC and ver == ECV3_VERSION, f"Bad header {magic!r} v{ver}"
    pos = 32
    tpls: dict[int, tuple[bytes, int, int]] = {}
    for _ in range(tpl_cnt):
        tid, tlen, ioff, moff = struct.unpack_from("<HHHH", data, pos)
        pos += 8
        tbytes = data[pos : pos + tlen]
        pos += tlen
        pad = (-(pos - 32)) % 4
        pos += pad
        tpls[tid] = (tbytes, ioff, moff)

    entries = []
    for i in range(tile_cnt):
        entries.append(struct.unpack_from("<IIIII", data, pos + i * 20))

    # Decode up to sample_n evenly-spaced tiles across all templates present
    step = max(1, len(entries) // max(sample_n, 1))
    decoded_ok = 0
    for key, off, ln, meta, aux in entries[::step][:sample_n]:
        tid = (meta >> 24) & 0xFF
        tbytes, ioff, moff = tpls[tid]
        obu = data[off : off + ln]
        rebuilt = reassemble_avif(tbytes, ioff, moff, obu)
        if tid == TPL_HEIGHT10:
            with tempfile.TemporaryDirectory() as td:
                p_in = Path(td) / "t.avif"
                p_out = Path(td) / "t.png"
                p_in.write_bytes(rebuilt)
                subprocess.run(
                    ["avifdec", str(p_in), str(p_out)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                im = Image.open(p_out)
                assert im.size == (ts, ts), f"Bad height tile size {im.size}"
        else:
            im = Image.open(io.BytesIO(rebuilt))
            assert im.size == (ts, ts), f"Bad tile size {im.size}"
        decoded_ok += 1

    return {
        "pack_id": pack_id,
        "tiles": tile_cnt,
        "templates": sorted(tpls.keys()),
        "bytes": len(data),
        "verified_decoded": decoded_ok,
    }

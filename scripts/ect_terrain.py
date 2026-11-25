"""Shared ECT1/ECT2 terrain encoding & decoding helpers for Earth Connect asset scripts."""

from __future__ import annotations

import gzip
import struct
import subprocess
from pathlib import Path

import numpy as np

ECT2_MAGIC = 0x45435432
ECT2_FLAG_DELTA_SHUFFLE = 1
ECT2_FLAG_COAST = 4


def _delta_zigzag_u16(codes: np.ndarray) -> bytes:
    pred = np.empty_like(codes, dtype=np.int32)
    pred[0, 0] = 32768
    pred[0, 1:] = codes[0, :-1]
    pred[1:, 0] = codes[:-1, 0]
    pred[1:, 1:] = (
        codes[1:, :-1].astype(np.int32) + codes[:-1, 1:].astype(np.int32)
    ) >> 1
    delta = (codes.astype(np.int32) - pred) & 0xFFFF
    signed = np.where(delta >= 32768, delta - 65536, delta).astype(np.int32)
    zz = ((signed << 1) ^ (signed >> 31)) & 0xFFFF
    return (zz >> 8).astype(np.uint8).tobytes() + (zz & 0xFF).astype(np.uint8).tobytes()


def _delta_zigzag_u8(channel: np.ndarray) -> bytes:
    pred = np.empty_like(channel, dtype=np.int32)
    pred[0, 0] = 128
    pred[0, 1:] = channel[0, :-1]
    pred[1:, 0] = channel[:-1, 0]
    pred[1:, 1:] = (
        channel[1:, :-1].astype(np.int32) + channel[:-1, 1:].astype(np.int32)
    ) >> 1
    delta = (channel.astype(np.int32) - pred) & 0xFF
    signed = np.where(delta >= 128, delta - 256, delta).astype(np.int32)
    return (((signed << 1) ^ (signed >> 31)) & 0xFF).astype(np.uint8).tobytes()


def _undelta_zigzag_u8(buf: bytes, height: int, width: int) -> np.ndarray:
    zz = np.frombuffer(buf, dtype=np.uint8).reshape(height, width).astype(np.int32)
    signed = (zz >> 1) ^ -(zz & 1)
    out = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        prev_row = out[y - 1] if y > 0 else None
        left = 128
        for x in range(width):
            pred = (
                (128 if x == 0 else left)
                if prev_row is None
                else (int(prev_row[0]) if x == 0 else (left + int(prev_row[x])) >> 1)
            )
            code = (pred + int(signed[y, x])) & 0xFF
            left = code
            out[y, x] = code
    return out


def encode_ect2_with_coast(
    code: np.ndarray, coast_u8: np.ndarray, zstd_level: int = 19
) -> bytes:
    """Encode uint16 (elevation+32768) + uint8 coast distance as lossless ECT2+coast (flags=5)."""
    height, width = code.shape
    payload = _delta_zigzag_u16(code) + _delta_zigzag_u8(coast_u8)
    proc = subprocess.run(
        [
            "zstd",
            f"-{zstd_level}",
            "-T0",
            "--content-size",
            f"--stream-size={len(payload)}",
            "-q",
            "-c",
        ],
        input=payload,
        capture_output=True,
        check=True,
    )
    header = struct.pack(">I", ECT2_MAGIC) + struct.pack(
        "<III", width, height, ECT2_FLAG_DELTA_SHUFFLE | ECT2_FLAG_COAST
    )
    return header + proc.stdout


def load_earth_coast(terrain_bin: Path) -> np.ndarray:
    """Decode (height, width) float32 signed coast distance (>0 inland, <0 ocean) from ECT2 or legacy ECT1."""
    raw = terrain_bin.read_bytes()
    if len(raw) >= 16 and struct.unpack_from(">I", raw, 0)[0] == ECT2_MAGIC:
        width, height, flags = struct.unpack_from("<III", raw, 4)
        if not (flags & ECT2_FLAG_COAST):
            raise ValueError(
                f"{terrain_bin} is ECT2 without a coast channel (flags={flags})"
            )
        payload = subprocess.run(
            ["zstd", "-d", "-q", "-c"], input=raw[16:], capture_output=True, check=True
        ).stdout
        count = width * height
        if flags & ECT2_FLAG_DELTA_SHUFFLE:
            coast_u8 = _undelta_zigzag_u8(payload[count * 2 : count * 3], height, width)
        else:
            coast_u8 = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)[
                :, :, 2
            ]
        return (coast_u8.astype(np.float32) - 128.0) / 4.0
    unpacked = gzip.decompress(raw)
    if unpacked[:4] != b"ECT1":
        raise ValueError(
            f"Unexpected terrain header in {terrain_bin}: {unpacked[:4]!r}"
        )
    width, height = struct.unpack_from("<II", unpacked, 4)
    rgba = np.frombuffer(unpacked, dtype=np.uint8, offset=12).reshape(height, width, 4)
    return (rgba[:, :, 2].astype(np.float32) - 128.0) / 4.0

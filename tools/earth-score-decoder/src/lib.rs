//! The worker owns this instance, its reusable Zstd context, and its WASM allocations.
use std::alloc::{Layout, alloc, dealloc};

static mut DCTX: *mut zstd_sys::ZSTD_DCtx = std::ptr::null_mut();

#[unsafe(no_mangle)]
pub extern "C" fn allocate(bytes: usize) -> *mut u8 {
    let Ok(layout) = Layout::from_size_align(bytes, 8) else {
        return std::ptr::null_mut();
    };
    if bytes == 0 {
        return std::ptr::null_mut();
    }
    unsafe { alloc(layout) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() {
        unsafe { dealloc(ptr, Layout::from_size_align(bytes, 8).unwrap()) };
    }
}

/// Returns zero on success. Both buffers must be valid for their given lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn decode(
    frame: *const u8,
    frame_bytes: usize,
    output: *mut u8,
    output_bytes: usize,
) -> u32 {
    unsafe {
        if zstd_sys::ZSTD_getFrameContentSize(frame.cast(), frame_bytes) != output_bytes as u64 {
            return 1;
        }
        let mut dctx = DCTX;
        if dctx.is_null() {
            dctx = zstd_sys::ZSTD_createDCtx();
            if dctx.is_null() {
                return 2;
            }
            DCTX = dctx;
        }
        let count = zstd_sys::ZSTD_decompressDCtx(
            dctx,
            output.cast(),
            output_bytes,
            frame.cast(),
            frame_bytes,
        );
        if zstd_sys::ZSTD_isError(count) != 0 || count != output_bytes {
            return 2;
        }
    }
    0
}

/// Decompresses a Zstd ECT2 payload and decodes byte-shuffled zigzag 2D-predicted
/// elevation (+ optional coast distance when `flags & 4 != 0`) directly into `output`.
/// `output` must hold `width * height * (4 if flags & 4 else 2)` bytes.
/// Returns 0 on success, 1 on frame content size mismatch, 2 on Zstd error, 3 on invalid dimensions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn decode_ect2(
    frame: *const u8,
    frame_bytes: usize,
    width: u32,
    height: u32,
    flags: u32,
    output: *mut u8,
) -> u32 {
    if width == 0 || height == 0 || width > 16384 || height > 16384 || output.is_null() {
        return 3;
    }
    let w = width as usize;
    let h = height as usize;
    let count = w * h;
    let has_coast = (flags & 4) != 0;
    let payload_stride: usize = if has_coast { 3 } else { 2 };
    let payload_bytes = count * payload_stride;

    let mut payload = vec![0u8; payload_bytes];
    let status = unsafe { decode(frame, frame_bytes, payload.as_mut_ptr(), payload_bytes) };
    if status != 0 {
        return status;
    }

    let out_stride: usize = if has_coast { 4 } else { 2 };
    let out = unsafe { std::slice::from_raw_parts_mut(output, count * out_stride) };

    if (flags & 1) != 0 {
        let mut prev_row = vec![0u16; w];
        let hi_plane = &payload[0..count];
        let lo_plane = &payload[count..count * 2];
        if has_coast {
            let mut prev_coast = vec![0u8; w];
            let coast_plane = &payload[count * 2..count * 3];
            // Row 0
            {
                let mut left: u32 = 32768;
                let mut left_coast: u32 = 128;
                for x in 0..w {
                    let zz = ((hi_plane[x] as u32) << 8) | (lo_plane[x] as u32);
                    let signed = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
                    let code = left.wrapping_add(signed as u32) & 0xffff;
                    left = code;
                    prev_row[x] = code as u16;

                    let czz = coast_plane[x] as u32;
                    let cs = ((czz >> 1) as i32) ^ -((czz & 1) as i32);
                    let ccode = left_coast.wrapping_add(cs as u32) & 0xff;
                    left_coast = ccode;
                    prev_coast[x] = ccode as u8;

                    let o = x * 4;
                    out[o] = (code >> 8) as u8;
                    out[o + 1] = (code & 0xff) as u8;
                    out[o + 2] = ccode as u8;
                    out[o + 3] = 255;
                }
            }
            // Rows 1..h
            for y in 1..h {
                let row_off = y * w;
                let hi_row = &hi_plane[row_off..row_off + w];
                let lo_row = &lo_plane[row_off..row_off + w];
                let c_row = &coast_plane[row_off..row_off + w];
                let out_row = &mut out[row_off * 4..(row_off + w) * 4];

                // x == 0
                let zz0 = ((hi_row[0] as u32) << 8) | (lo_row[0] as u32);
                let s0 = ((zz0 >> 1) as i32) ^ -((zz0 & 1) as i32);
                let mut left = (prev_row[0] as u32).wrapping_add(s0 as u32) & 0xffff;
                prev_row[0] = left as u16;

                let czz0 = c_row[0] as u32;
                let cs0 = ((czz0 >> 1) as i32) ^ -((czz0 & 1) as i32);
                let mut left_coast = (prev_coast[0] as u32).wrapping_add(cs0 as u32) & 0xff;
                prev_coast[0] = left_coast as u8;

                out_row[0] = (left >> 8) as u8;
                out_row[1] = (left & 0xff) as u8;
                out_row[2] = left_coast as u8;
                out_row[3] = 255;

                for x in 1..w {
                    let zz = ((hi_row[x] as u32) << 8) | (lo_row[x] as u32);
                    let signed = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
                    let pred = (left + prev_row[x] as u32) >> 1;
                    let code = pred.wrapping_add(signed as u32) & 0xffff;
                    left = code;
                    prev_row[x] = code as u16;

                    let czz = c_row[x] as u32;
                    let cs = ((czz >> 1) as i32) ^ -((czz & 1) as i32);
                    let cpred = (left_coast + prev_coast[x] as u32) >> 1;
                    let ccode = cpred.wrapping_add(cs as u32) & 0xff;
                    left_coast = ccode;
                    prev_coast[x] = ccode as u8;

                    let o = x * 4;
                    out_row[o] = (code >> 8) as u8;
                    out_row[o + 1] = (code & 0xff) as u8;
                    out_row[o + 2] = ccode as u8;
                    out_row[o + 3] = 255;
                }
            }
        } else {
            // Stride 2 (no coast)
            // Row 0
            {
                let mut left: u32 = 32768;
                for x in 0..w {
                    let zz = ((hi_plane[x] as u32) << 8) | (lo_plane[x] as u32);
                    let signed = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
                    let code = left.wrapping_add(signed as u32) & 0xffff;
                    left = code;
                    prev_row[x] = code as u16;
                    let o = x * 2;
                    out[o] = (code >> 8) as u8;
                    out[o + 1] = (code & 0xff) as u8;
                }
            }
            // Rows 1..h
            for y in 1..h {
                let row_off = y * w;
                let hi_row = &hi_plane[row_off..row_off + w];
                let lo_row = &lo_plane[row_off..row_off + w];
                let out_row = &mut out[row_off * 2..(row_off + w) * 2];

                let zz0 = ((hi_row[0] as u32) << 8) | (lo_row[0] as u32);
                let s0 = ((zz0 >> 1) as i32) ^ -((zz0 & 1) as i32);
                let mut left = (prev_row[0] as u32).wrapping_add(s0 as u32) & 0xffff;
                prev_row[0] = left as u16;
                out_row[0] = (left >> 8) as u8;
                out_row[1] = (left & 0xff) as u8;

                for x in 1..w {
                    let zz = ((hi_row[x] as u32) << 8) | (lo_row[x] as u32);
                    let signed = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
                    let pred = (left + prev_row[x] as u32) >> 1;
                    let code = pred.wrapping_add(signed as u32) & 0xffff;
                    left = code;
                    prev_row[x] = code as u16;
                    let o = x * 2;
                    out_row[o] = (code >> 8) as u8;
                    out_row[o + 1] = (code & 0xff) as u8;
                }
            }
        }
    } else if has_coast {
        for i in 0..count {
            let s = i * 3;
            let o = i * 4;
            out[o] = payload[s];
            out[o + 1] = payload[s + 1];
            out[o + 2] = payload[s + 2];
            out[o + 3] = 255;
        }
    } else {
        out.copy_from_slice(&payload);
    }

    0
}

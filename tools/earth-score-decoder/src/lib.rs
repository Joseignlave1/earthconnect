//! The worker owns this instance and the two disjoint WASM allocations.
use std::alloc::{Layout, alloc, dealloc};

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
        let count =
            zstd_sys::ZSTD_decompress(output.cast(), output_bytes, frame.cast(), frame_bytes);
        if zstd_sys::ZSTD_isError(count) != 0 || count != output_bytes {
            return 2;
        }
    }
    0
}

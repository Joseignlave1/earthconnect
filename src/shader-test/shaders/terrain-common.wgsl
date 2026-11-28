const EARTH_PI: f32 = 3.141592653589793;
const EARTH_EQUATORIAL_RADIUS_M: f32 = 6378137.0;

fn wrapClampTexel(pixel: vec2i, size: vec2i) -> vec2i {
  return vec2i(((pixel.x % size.x) + size.x) % size.x, clamp(pixel.y, 0, size.y - 1));
}

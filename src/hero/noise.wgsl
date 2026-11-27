fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let s = f * f * (3. - 2. * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1., 0.));
  let c = hash21(i + vec2f(0., 1.));
  let d = hash21(i + vec2f(1., 1.));
  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

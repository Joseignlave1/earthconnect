fn toLinear(c: vec3f) -> vec3f {
  return mix(c / 12.92, pow((c + .055) / 1.055, vec3f(2.4)), step(vec3f(.04045), c));
}

fn toDisplay(c: vec3f) -> vec3f {
  return mix(c * 12.92, 1.055 * pow(max(c, vec3f(0.)), vec3f(1. / 2.4)) - .055, step(vec3f(.0031308), c));
}

fn acesToneMap(c: vec3f) -> vec3f {
  return clamp((c * (2.51 * c + .03)) / (c * (2.43 * c + .59) + .14), vec3f(0.), vec3f(1.));
}

// Piecewise degree-4 polynomial fit to Magma (C0-continuous at t = 0.5, max error ~0.01):
// eliminates the dynamically indexed 384-byte stack array that spilled registers in surfaceFragment.
fn magma(value: f32) -> vec3f {
  let t = clamp(value, 0., 1.);
  if (t < .5) {
    let u = t * 2.;
    let c = vec3f(.00113, -.00571, .00618) + u * (vec3f(.08028, .51309, .92460) + u * (vec3f(2.00634, -1.61980, 1.60430) + u * (vec3f(-2.30685, 2.41181, -4.14306) + u * vec3f(.93190, -1.08539, 2.08468))));
    return clamp(c, vec3f(0.), vec3f(1.));
  }
  let u = t * 2. - 1.;
  let c = vec3f(.71280, .21400, .47670) + u * (vec3f(.92890, .14288, -.25896) + u * (vec3f(-.63725, 1.40238, -1.13907) + u * (vec3f(-.46885, -.95816, 3.39635) + u * vec3f(.45417, .18874, -1.72973))));
  return clamp(c, vec3f(0.), vec3f(1.));
}

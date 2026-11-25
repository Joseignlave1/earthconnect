fn pulseIntegral(x: f32, d: f32) -> f32 {
  return d * floor(x) + min(fract(x), d);
}

// Exact analytical box integral of a periodic pulse train (pulses of width d centered at integers)
// over local-phase offsets [aL, bL] relative to integer pulse index cell, normalized by full pixel
// filter width. Keeping endpoints relative to cell preserves full f32 precision on flat plateaus.
fn filteredPulseLocal(cell: f32, aL: f32, bL: f32, fullStep: f32, widthPx: f32, aaPx: f32, maxDuty: f32, skipZero: f32) -> f32 {
  if (bL <= aL) { return 0.; }
  let w = max(fullStep, 1e-6);
  let rawD = widthPx * w;
  let knee = maxDuty * .70;
  let headroom = max(1e-4, maxDuty - knee);
  let d = select(rawD, knee + headroom * (1. - exp(-(rawD - knee) / headroom)), rawD > knee);
  let fw = max(w * aaPx, 1e-6);
  if (fw < .25 && d < .25) {
    if (skipZero > .5 && abs(cell) < .5) { return 0.; }
    return clamp(max(0., min(bL, .5 * d) - max(aL, -.5 * d)) / fw, 0., 1.);
  }
  let a = aL + .5 * d;
  let b = bL + .5 * d;
  var coverage = (pulseIntegral(b, d) - pulseIntegral(a, d)) / fw;
  if (skipZero > .5) {
    let z0 = -cell;
    let zeroOverlap = max(0., min(b, z0 + d) - max(a, z0)) / fw;
    coverage = max(0., coverage - zeroOverlap);
  }
  return clamp(coverage, 0., 1.);
}

fn filteredPulse(coord: f32, stepPerPx: f32, widthPx: f32, aaPx: f32, maxDuty: f32, skipZero: f32) -> f32 {
  let cell = floor(coord + .5);
  let local = coord - cell;
  let fw = max(max(stepPerPx, 1e-6) * aaPx, 1e-6);
  return filteredPulseLocal(cell, local - .5 * fw, local + .5 * fw, stepPerPx, widthPx, aaPx, maxDuty, skipZero);
}

fn contourLine(height: f32, gradient: vec2f, interval: f32, width: f32, pixelRatio: f32, skipZero: f32) -> f32 {
  // On Earth, zero belongs to the geographic coastline, including on the seabed.
  let aaPx = mix(1., .7, smoothstep(1., 1.8, pixelRatio));
  return filteredPulse(height / interval, length(gradient) / interval, width * pixelRatio, aaPx, .58, skipZero);
}

fn contourLineSplit(height: f32, gradLen: f32, landInterval: f32, seaInterval: f32, width: f32, pixelRatio: f32) -> f32 {
  let aaPx = mix(1., .7, smoothstep(1., 1.8, pixelRatio));
  let halfSpan = .5 * max(gradLen, 1e-6) * aaPx;
  let h0 = height - halfSpan;
  let h1 = height + halfSpan;
  if (h0 < 0. && h1 > 0. && abs(landInterval - seaInterval) > .5) {
    let landB = h1 / landInterval;
    let landCell = floor(.5 * landB + .5);
    let landCov = filteredPulseLocal(landCell, -landCell, landB - landCell, gradLen / landInterval, width * pixelRatio, aaPx, .58, 1.);
    let seaA = h0 / seaInterval;
    let seaCell = floor(.5 * seaA + .5);
    let seaCov = filteredPulseLocal(seaCell, seaA - seaCell, -seaCell, gradLen / seaInterval, width * pixelRatio, aaPx, .58, 1.);
    return clamp(landCov + seaCov, 0., 1.);
  }
  let interval = select(landInterval, seaInterval, height < 0.);
  return filteredPulse(height / interval, gradLen / interval, width * pixelRatio, aaPx, .58, 1.);
}

fn terrainContours(height: f32, gradient: vec2f, interval: f32, detail: vec2f, pixelRatio: f32, skipZero: f32) -> f32 {
  let major = contourLine(height, gradient, interval, 1.25, pixelRatio, skipZero);
  let minor = contourLine(height, gradient, interval * .5, .95, pixelRatio, skipZero) * detail.x * .65;
  let fine = contourLine(height, gradient, interval * .25, .8, pixelRatio, skipZero) * detail.y * .4;
  return max(major, max(minor, fine));
}

fn terrainContoursSplit(height: f32, gradient: vec2f, landInterval: f32, seaInterval: f32, detail: vec2f, pixelRatio: f32) -> f32 {
  let g = length(gradient);
  let major = contourLineSplit(height, g, landInterval, seaInterval, 1.25, pixelRatio);
  let minor = contourLineSplit(height, g, landInterval * .5, seaInterval * .5, .95, pixelRatio) * detail.x * .65;
  let fine = contourLineSplit(height, g, landInterval * .25, seaInterval * .25, .8, pixelRatio) * detail.y * .4;
  return max(major, max(minor, fine));
}

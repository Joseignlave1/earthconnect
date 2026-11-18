// Whole contour families share these weights. Local terrain slope must not
// decide visibility: doing that erases pieces of otherwise continuous lines.
export function contourDetail(metresPerPixel: number): [number, number] {
  const reveal = (near: number, far: number) => {
    const t = Math.max(0,Math.min(1,(metresPerPixel - near) / (far - near)))
    return 1 - t * t * (3 - 2 * t)
  }
  return [reveal(4000,12000),reveal(1000,4000)]
}

export const contourShader = `
fn contourLine(height: f32,gradient: vec2f,interval: f32,width: f32) -> f32 {
  let level = floor(height / interval + .5);
  let distance = abs(height - level * interval) / max(.000001,length(gradient));
  let radius = width * u.contour.w * .5;
  // Zero belongs to the geographic coastline, including on the seabed.
  let aa = .35;
  return step(1.,abs(level)) * (1. - smoothstep(radius - aa,radius + aa,distance));
}
fn terrainContours(height: f32,gradient: vec2f,interval: f32,detail: vec2f) -> f32 {
  let major = contourLine(height,gradient,interval,1.25);
  let minor = contourLine(height,gradient,interval * .5,.95) * detail.x * .65;
  let fine = contourLine(height,gradient,interval * .25,.8) * detail.y * .4;
  return max(major,max(minor,fine));
}`

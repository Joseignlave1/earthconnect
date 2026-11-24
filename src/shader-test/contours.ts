// Whole contour families share these weights. Local terrain slope must not
// decide visibility: doing that erases pieces of otherwise continuous lines.
export function contourDetail(metresPerPixel: number): [number, number] {
  const reveal = (near: number, far: number) => {
    const t = Math.max(0, Math.min(1, (metresPerPixel - near) / (far - near)))
    return 1 - t * t * (3 - 2 * t)
  }
  return [reveal(4000, 12000), reveal(1000, 4000)]
}

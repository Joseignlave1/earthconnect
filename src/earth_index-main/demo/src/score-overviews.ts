export interface ScoreLevel {
  width: number;
  height: number;
  scores: Uint16Array;
}

// Every overview covers the original extent. Pool directly from native cells so
// odd dimensions neither pad the footprint nor accumulate shifts between levels.
export function scorePyramid(scores: Uint16Array, width: number, height: number): ScoreLevel[] {
  const levels: ScoreLevel[] = [{ width, height, scores }];
  let w = width,
    h = height;
  while (w > 1 || h > 1) {
    w = Math.ceil(w / 2);
    h = Math.ceil(h / 2);
    const pooled = new Uint16Array(w * h).fill(65535);
    for (let y = 0; y < height; y++) {
      const row = Math.min(h - 1, Math.floor(((y + 0.5) * h) / height)) * w;
      for (let x = 0; x < width; x++) {
        const score = scores[y * width + x];
        if (score === 65535) continue;
        const index = row + Math.min(w - 1, Math.floor(((x + 0.5) * w) / width));
        if (pooled[index] === 65535 || score > pooled[index]) pooled[index] = score;
      }
    }
    levels.push({ width: w, height: h, scores: pooled });
  }
  return levels;
}

export function overviewLevel(levels: ScoreLevel[], span: number[], resolution: number) {
  if (!(resolution > 0)) return 0;
  // Choose texels at least one screen pixel across, so nearest-neighbour
  // reprojection cannot sample past an isolated native peak on zoom-out.
  const index = levels.findIndex(
    (level) => Math.min(span[0] / level.width, span[1] / level.height) >= resolution,
  );
  return index < 0 ? levels.length - 1 : index;
}

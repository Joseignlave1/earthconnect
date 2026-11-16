import type { AreaPatch, Area } from './types.ts';
// Score-space controls operate on the received matrix; no second SQL query.
export const SCORE_MAX = 65534;
export const SCORE_MISSING = 65535;
export function thresholdCode(percent: number) {
  if (!Number.isFinite(percent) || percent < 0 || percent > 100)
    throw new Error('Score threshold must be between 0 and 100');
  return Math.ceil((percent * SCORE_MAX) / 100);
}
export function scoreDistribution(patches: Pick<AreaPatch, 'scores'>[], bands = 50) {
  const bins = new Uint32Array(bands);
  let valid = 0,
    missing = 0,
    zero = 0,
    best = 0;
  for (const { scores } of patches)
    for (const score of scores) {
      if (score === SCORE_MISSING) {
        missing++;
        continue;
      }
      valid++;
      if (score === 0) zero++;
      best = Math.max(best, score);
      bins[Math.min(bands - 1, Math.floor((score / SCORE_MAX) * bands))]++;
    }
  return { bins, valid, missing, zero, best };
}
export function findAreas(patches: AreaPatch[], cutoff: number, limit = 50) {
  let count = 0,
    areaCount = 0;
  const top: Area[] = [];
  const compare = (a: Area, b: Area) =>
    b.peak - a.peak ||
    b.mean - a.mean ||
    b.count - a.count ||
    a.siteKey.localeCompare(b.siteKey) ||
    a.peakIndex - b.peakIndex;
  for (const { grid, scores } of patches) {
    const { width, height, siteKey } = grid;
    const visited = new Uint8Array(scores.length);
    const stack = new Int32Array(scores.length);
    const matches = (i: number) =>
      scores[i] !== SCORE_MISSING && scores[i] > 0 && scores[i] >= cutoff;
    for (let seed = 0; seed < scores.length; seed++) {
      if (visited[seed] || !matches(seed)) continue;
      let size = 1;
      stack[0] = seed;
      visited[seed] = 1;
      const area: Area = {
        siteKey,
        count: 0,
        peak: 0,
        peakIndex: seed,
        mean: 0,
        minX: width,
        maxX: 0,
        minY: height,
        maxY: 0,
      };
      let sum = 0;
      while (size) {
        const i = stack[--size],
          x = i % width,
          y = Math.floor(i / width),
          score = scores[i];
        area.count++;
        sum += score;
        if (score > area.peak || (score === area.peak && i < area.peakIndex)) {
          area.peak = score;
          area.peakIndex = i;
        }
        area.minX = Math.min(area.minX, x);
        area.maxX = Math.max(area.maxX, x);
        area.minY = Math.min(area.minY, y);
        area.maxY = Math.max(area.maxY, y);
        for (const neighbor of [
          x > 0 ? i - 1 : -1,
          x + 1 < width ? i + 1 : -1,
          y > 0 ? i - width : -1,
          y + 1 < height ? i + width : -1,
        ]) {
          if (neighbor >= 0 && !visited[neighbor] && matches(neighbor)) {
            visited[neighbor] = 1;
            stack[size++] = neighbor;
          }
        }
      }
      count += area.count;
      areaCount++;
      area.mean = sum / area.count;
      // Most isolated cells never enter the shortlist; avoid sorting fifty
      // entries for every island in a noisy matrix.
      if (top.length < limit || compare(area, top[top.length - 1]) < 0) {
        let low = 0,
          high = top.length;
        while (low < high) {
          const middle = (low + high) >>> 1;
          if (compare(area, top[middle]) < 0) high = middle;
          else low = middle + 1;
        }
        top.splice(low, 0, area);
        if (top.length > limit) top.pop();
      }
    }
  }
  return { count, areaCount, top };
}

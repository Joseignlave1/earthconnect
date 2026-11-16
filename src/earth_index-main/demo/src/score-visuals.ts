import { findAreas } from './picks.ts';
import type { AreaPatch } from './types.ts';

export const SCORE_PURPLE = [168, 85, 247] as const;

// One scale for the entire result, kept stable while the visibility cutoff changes.
export function scoreIntensity(score: number, maximum: number) {
  if (score <= 0 || score === 65535 || maximum <= 0) return 0;
  return Math.min(1, score / maximum) ** 1.6;
}

export function strongestAreas(patches: AreaPatch[], maximum: number) {
  if (maximum <= 0) return [];
  // Tiny highlights represent area peaks, never every cell or a patch centroid.
  return findAreas(patches, Math.ceil(maximum * 0.9), 100).top;
}

export function glowOpacity(resolution: number, areaSpan: number, relative: number) {
  if (resolution <= 0 || areaSpan <= 0) return 0;
  // Full strength for areas <=2 px across, gone at 4 px: visible raster
  // shapes take over long before individual native cells can be inspected.
  const fade = Math.max(0, Math.min(1, (resolution / areaSpan - 0.25) / 0.25));
  return fade * Math.min(1, Math.max(0, relative)) ** 2;
}

export function purpleCss(intensity: number) {
  return `rgba(${SCORE_PURPLE.join(',')},${Math.max(0, Math.min(1, intensity))})`;
}

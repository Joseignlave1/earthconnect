import type { CoverageMetadata } from './types.ts';
import Polygon from 'ol/geom/Polygon.js';
import MultiPolygon from 'ol/geom/MultiPolygon.js';
import WKB from 'ol/format/WKB.js';

export function coverageFeatures(datasource: CoverageMetadata) {
  return datasource.coverage.flatMap((patch) => {
    if (patch.srid !== 4326) throw new Error('Unsupported coverage coordinate system');
    const features = new WKB().readFeatures(patch.wkb);
    for (const feature of features) {
      if (
        !(feature.getGeometry() instanceof Polygon) &&
        !(feature.getGeometry() instanceof MultiPolygon)
      )
        throw new Error('Invalid coverage geometry');
      feature.set('siteKey', patch.siteKey);
    }
    return features;
  });
}

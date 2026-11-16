#!/usr/bin/env python3
"""Compile the reviewed point-source registry; no network or database access.

Only successfully audited numeric/feature endpoints are enabled. Unknown units
stay raw and carry semantic_review_required. Terrain window statistics require
an area sampler and are intentionally not substituted by a point measurement.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'research/planetary-sources'
catalog = json.loads((RESEARCH / 'catalog.json').read_text())
trek = json.loads((RESEARCH / 'trek-audit.json').read_text())
metadata = json.loads((RESEARCH / 'mars-source-metadata.json').read_text())['layers']
services = {s['url']: s for s in trek['services']}
sources = []


def projection(sr):
    wkt = sr.get('wkt', '')
    def parameter(name):
        match = re.search(r'PARAMETER\["' + name + r'",([\d.eE+-]+)\]', wkt, re.I)
        assert match, (name, wkt)
        return float(match[1])
    if not wkt.startswith('PROJCS'):
        return {'kind': 'geographic'}
    if 'Stereographic_North_Pole' in wkt:
        return {'kind': 'polar', 'north': True}
    if 'Stereographic_South_Pole' in wkt:
        return {'kind': 'polar', 'north': False}
    if 'Orthographic' in wkt:
        return {'kind': 'orthographic', 'longitude': parameter('Longitude_Of_Center'), 'latitude': parameter('Latitude_Of_Center')}
    if 'Plate_Carree' in wkt:
        return {'kind': 'plate_carree'}
    raise ValueError('Unsupported native projection: ' + wkt)


for variable in catalog['variables']:
    if not variable['readiness'].startswith('verified_transport'):
        continue
    seen = set()
    for evidence in variable['sources']:
        if evidence['status'] not in ['numeric_response_verified', 'numeric_cutout_verified', 'spatial_feature_query_verified']:
            continue
        name = evidence['service']
        # Duplicate Trek aliases for the same product are not extra measurements.
        if name in seen:
            continue
        seen.add(name)
        flags = []
        unit = variable['unit']
        if variable['readiness'].endswith('semantic_review'):
            flags += ['semantic_review_required', 'raw_source_scale']
            unit = 'raw'
        if variable['earth_comparison'] != 'comparable_after_support_alignment':
            flags += ['cross_body_comparison_requires_review']
        source = dict(body=variable['body'], variable_id=variable['id'], label=variable['label'],
                      unit=unit, notes=variable['notes'], quality_flags=flags, source_id=name)
        if evidence['provider'] == 'NASA Trek':
            service = services[evidence['url']]
            m = service['metadata']['data']
            extent = m.get('extent', m.get('fullExtent'))
            sr = m.get('spatialReference', extent.get('spatialReference', {}))
            feature = service['type'] == 'MapServer'
            source.update(kind='trek_feature' if feature else 'trek_raster', url=evidence['url'],
                          projection=projection(sr), extent=[extent[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']],
                          pixel_spacing=m.get('pixelSizeX', 0),
                          pixel_spacing_unit='m' if sr.get('wkt', '').startswith('PROJCS') else 'degree',
                          nodata=[-32768, -99999])
            if feature:
                source['layer_id'] = service['feature_layer']['data']['id']
                source['unit'] = 'source_attributes'
                source['notes'] += ' Value is a list of source attribute objects intersecting the exact point; no proximity search or absence inference.'
            if variable['id'] in ['moon.solar_illumination', 'moon.earth_visibility']:
                source['valid_range'] = [0, 1]
            if variable['id'].startswith('moon.surface_'):
                source['valid_range'] = [0.01, 500]
            if variable['id'] == 'moon.christiansen_feature':
                source['valid_range'] = [5, 12]
                source['quality_flags'] += ['conservative_physical_range_mask']
        else:
            keywords = metadata.get(name, {}).get('keywords', [])
            ppd = next((float(k.split(':', 1)[1]) for k in keywords if k.startswith('maxPPD:')), 4.)
            if name.startswith('mars_yearly_'):
                ppd = 8.  # The reviewed KRC extrema use the same grid as the mean.
            nodata = [float(k.split(':', 1)[1]) for k in keywords if k.startswith('NullValue:')]
            if name.startswith('TES_') and name.endswith('_numeric') and variable['id'].startswith('mars.tes_'):
                nodata += [-1]
            if name == 'TES_TI_Putzig_night_numeric':
                nodata += [-32768]
            source.update(kind='mars_wms', url='https://ms-mars.mars.asu.edu/alb32', ppd=ppd,
                          projection={'kind': 'geographic'}, extent=[-180, -90, 180, 90],
                          pixel_spacing=1/ppd, pixel_spacing_unit='degree', nodata=nodata)
            if 'temperature' in variable['id']:
                source['quality_flags'] += ['modeled_surface_temperature']
        sources.append(source)

result = {'audit_date': '2026-11-15', 'sources': sources}
path = ROOT / 'backend/src/planetary/sources.json'
path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'sources': len(sources), 'moon': sum(s['body'] == 'moon' for s in sources), 'mars': sum(s['body'] == 'mars' for s in sources)}))

# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "rasterio==1.5.2", "numpy==2.5.4", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Independent NumPy score fixtures on three global regions, six frozen targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from load_raster_pilot import ROOT
from validate_expansion import score


def main():
    out=ROOT/'research/world-4km'
    directory=out/'validation'
    directory.mkdir(exist_ok=True)
    data=json.loads((out/'manifest.json').read_text())
    targets=json.loads((ROOT/'research/expansion-4km/validation/targets.json').read_text())
    sites=['h11v11','h08v05','h15v01']
    cases=[]
    for site in sites:
        values={}
        for layer in data['layers']:
            if layer['site_key']!=site:continue
            with rasterio.open(ROOT/layer['path']) as src:a=src.read(1)
            values[layer['feature_key']]=np.where(a==0,np.nan,a.astype(float)*layer['scale']+layer['offset'])
        for target in targets:
            codes=score(values,target)
            name=f"{target['target']['key']}--{site}.bin"
            payload=codes.tobytes()
            (directory/name).write_bytes(payload)
            cases.append(dict(target=target['target']['key'],site_key=site,path=name,
                              sha256=hashlib.sha256(payload).hexdigest(),cells=codes.size,
                              known_cells=int((codes!=65535).sum())))
    (directory/'api-cases.json').write_text(json.dumps(dict(sites=sites,cases=cases),indent=2)+'\n')
    print('Prepared',len(cases),'independent packed score fixtures;',sum(c['cells'] for c in cases),'score codes')


if __name__=='__main__':main()

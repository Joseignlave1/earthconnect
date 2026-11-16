"""Build the inline demonstration from measured results, using only stdlib."""
import array
import base64
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/k2-demo'
grid=json.loads((OUT/'grid.json').read_text())
results=json.loads((OUT/'results.json').read_text())
bands={}
decoded={}
for name,meta in grid['layers'].items():
    raw=(OUT/f'{name}.u16').read_bytes()
    values=array.array('H');values.frombytes(raw)
    decoded[name]=[None if code==0 else (code+meta['integer_origin'])*meta['quantum'] for code in values]
    bands[name]={'base64':base64.b64encode(raw).decode(),'origin':meta['integer_origin'],
                 'quantum':meta['quantum'],'unit':meta['unit'],
                 'range':[meta['physical_min']-meta['quantum']/2,meta['physical_max']+meta['quantum']/2]}
tests=[]
for hot,flat in [(310,.9),(320,.95),(327,1)]:
    count=sum(t is not None and t+1e-8>=hot and c+1e-8>=.95 and f+1e-8>=flat
              for t,c,f in zip(decoded['day_temperature_k'],decoded['day_temperature_valid_fraction'],decoded['slope_lt_5deg_fraction']))
    tests.append({'hot':hot,'flat':flat,'count':count})
assert tests[0]['count']==results['queries']['hot_and_90pct_native_terrain_slope_lt_5']
totals=results['totals']
chart={'id':'k2-storage','title':'Storage for five 1 km bands','height':250,
       'description':'Same 50,000 quantized values; smaller is less storage.',
       'spec':{'type':'horizontalBar','y':'representation','x':'bytes','xLabel':'Bytes',
               'showValues':True,'showLegend':False,'sortOrder':'original'},
       'columns':['representation','bytes'],'rows':[
           ['Packed uint16',totals['raw16_bytes']],['k²-raster',totals['k2r_bytes']],
           ['Heuristic k²-raster',totals['k2h_bytes']],['zlib on uint16',totals['zlib16_bytes']]]}
html=(ROOT/'scripts/k2_demo_visual.template.html').read_text()
html=html.replace('__CHART__',json.dumps(chart)).replace('__DATA__',json.dumps({'bands':bands,'tests':tests},separators=(',',':')))
(OUT/'demo.html').write_text(html)
print(f'Wrote {OUT / "demo.html"}; {len(html.encode())} bytes; filters verified: {tests}')

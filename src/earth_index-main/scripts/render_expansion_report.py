"""Render the actual experiment comparison as a self-contained inline HTML artifact."""
import base64
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'research/expansion-4km'
report=json.loads((out/'validation-report.json').read_text())
run=json.loads((out/'manifest.json').read_text())
load=json.loads((out/'database.json').read_text())
plan=json.loads((out/'plan.json').read_text())
old=json.loads((out/'validation/baseline-matching.json').read_text())['results']
new=json.loads((out/'validation/expanded-matching.json').read_text())['results']
coverage=json.loads((out/'coverage.geojson').read_text())
oldcoverage=json.loads((ROOT/'research/pilots/database/current-coverage-map-data.json').read_text())

def path(geometry):
    return ' '.join('M'+' L'.join(f'{lon+180:.3f},{90-lat:.3f}' for lon,lat in ring[::max(1,len(ring)//40)])+' Z' for ring in geometry['coordinates'])

map_config=next(config for config in json.loads((ROOT/'demo/src/maps.json').read_text()) if config['id']=='earth')
span=map_config['tileSize']*map_config['resolutions'][0]
images=''.join(f'<image x="{index*span}" y="0" width="{span}" height="{span}" href="data:image/jpeg;base64,{base64.b64encode((ROOT/f"demo/public/overview/earth-{index}.jpg").read_bytes()).decode()}"/>' for index in (0,1))
newpaths=''.join(f'<path d="{path(f["geometry"])}"><title>{html.escape(f["properties"]["site_key"])}</title></path>' for f in coverage['features'])
oldpaths=''.join(f'<path d="{path(f["footprint"])}"/>' for f in oldcoverage['sites'])
forecast=plan['forecast']['predicted_seconds']/60
actual=(run['wall_seconds']+load['wall_seconds'])/60
stages=run['stage_seconds'];maximum=max(forecast,actual)
timebars=''.join(f'<div class="timing"><span>{label}</span><div class="track"><i style="width:{value/maximum*100:.2f}%;background:var(--{color})"></i></div><b>{value:.1f} min</b></div>' for label,value,color in [('Forecast',forecast,'chart-2'),('Measured import + DB',actual,'chart-1')])
rows=''
for baseline,expanded in zip(old,new):
    assert baseline['target']['key']==expanded['target']['key']
    a=baseline['patches'][0];b=expanded['patches'][0]
    rows+=f'<tr><td>{html.escape(expanded["target"]["name"])}</td><td>{a["max"]:.3f}</td><td>{b["max"]:.3f}<div class="scoretrack"><i style="width:{b["max"]*100:.1f}%"></i></div></td><td>{html.escape(b["siteKey"])}</td></tr>'
area=report['coverage']['union_area_km2']
text=f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:var(--font-sans);color:var(--foreground)}}p{{font-size:12px;line-height:1.5;color:var(--muted-foreground);margin:8px 0 14px}}h3{{font-size:15px;margin:18px 0 10px}}svg{{width:100%;display:block;border-radius:var(--radius)}}.timing{{display:grid;grid-template-columns:155px 1fr 70px;align-items:center;gap:10px;margin:10px 0;font-size:12px}}.track{{height:13px;background:var(--secondary);border-radius:3px;overflow:hidden}}.track i{{height:13px;display:block}}b{{text-align:right;font-variant-numeric:tabular-nums}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px 4px;text-align:left;border-bottom:1px solid var(--border)}}th{{font-weight:500;color:var(--muted-foreground)}}.scoretrack{{width:100%;height:4px;background:var(--secondary);margin-top:4px}}.scoretrack i{{height:4px;background:var(--chart-1);display:block}}.legend{{font-size:12px;display:flex;gap:20px;margin:9px 0}}.legend span:before{{content:'';display:inline-block;width:10px;height:10px;margin-right:6px;background:var(--chart-1)}}.legend span:last-child:before{{background:var(--chart-2)}}@media(max-width:500px){{.timing{{grid-template-columns:115px 1fr 62px;gap:6px;font-size:11px}}table{{font-size:11px}}}}
</style></head><body>{timebars}
<p>Cold source downloads, eight thermal download workers and eight CPU workers. Catalog {stages['catalog_seconds']/60:.1f} min · downloads {stages['download_seconds']/60:.1f} min · processing {stages['processing_seconds']/60:.1f} min · DB {load['wall_seconds']:.1f} s. Setup and later diagnostic analysis excluded.</p>
<svg viewBox="0 0 360 180" role="img" aria-label="Loaded Earth footprints"><g opacity=".6">{images}</g><g fill="var(--chart-1)" fill-opacity=".45" stroke="var(--chart-1)" stroke-width=".35">{newpaths}</g><g fill="var(--chart-2)" stroke="var(--chart-2)" stroke-width=".25">{oldpaths}</g></svg>
<div class="legend"><span>New 400 km regions</span><span>Original 100 km regions</span></div>
<p>{area/1e6:.2f} million km² of unique footprint ({report['coverage']['percent_earth_surface']:.2f}% of Earth). Footprints include water and missing evidence. 380,000 cells · 25 U16 variables · 4 × 4 km per cell.</p>
<h3>Same six targets, unchanged matching rules</h3><table><thead><tr><th>Target</th><th>Old best</th><th>New best</th><th>New best region</th></tr></thead><tbody>{rows}</tbody></table>
<p>Scores are relative similarity, not probabilities. More area also changes the available candidates; shared-area and subcell comparisons are recorded separately. Lunar thermal contrasts remain poorly matched on Earth.</p></body></html>'''
(out/'report.html').write_text(text)
print(out/'report.html')

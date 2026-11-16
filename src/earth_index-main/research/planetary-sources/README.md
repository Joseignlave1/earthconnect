# Moon/Mars target measurements: verified source audit

Checked 2026-11-14/15 UTC using anonymous public requests. We can obtain many
useful measurements for a clicked point or small window without downloading
global rasters. NASA Trek covers terrain, lunar thermal products, polar models,
composition and mapped features. ASU/JMARS supplies additional **numeric Mars
cutouts**, including thermal inertia, albedo and spectral-model abundances.

This report is source research. The backend now exposes the reviewed point
sources through [SamplePoint](../../backend/README.md#moonmars-point-measurements);
window statistics and the broader demo expansion remain separate. Numeric
transport verification does not certify a physical unit, a quality mask,
effective resolution or scientific accuracy.

## Results and scope

| Probe family | Result |
| --- | --- |
| NASA Trek representative endpoints | 91 checked: 68 numeric responses, 13 spatial feature queries |
| Trek unavailable/error | 2 metadata timeouts; 1 sample request rejected |
| Trek rendered imagery | 7 endpoints excluded from scientific measurements |
| ASU/JMARS tiny numeric cutouts | 22 of 25 layer-name probes decoded; 3 guessed names returned LayerNotDefined |
| Authentication | No token, cookies, key or login supplied |

These are endpoint counts, including alternate products/projections, not counts
of independent physical variables. The audit is a curated useful subset of the
discovered inventories, not every planetary dataset or regional DEM.
The three failed ASU names do not establish that the corresponding science
products are absent: names may differ.

The [full variable list](VARIABLES.md) has 99 concepts (56 Moon, 43 Mars), including
derived statistics, model alternatives and explicitly unresolved candidates.
The [machine-readable catalog](catalog.json) records sources, units, priorities,
coverage evidence and limitations for each concept. Its current readiness split:

| Status | Concepts | Meaning |
| --- | --- | --- |
| verified_transport | 47 | An appropriate numeric/spatial response was observed; source-specific quality checks still apply |
| verified_transport_requires_semantic_review | 20 | Numeric transport works; units, normalization or model semantics still need work |
| requires_local_derivation | 26 | Computable from verified inputs; calculation not implemented here |
| see_external_evidence_or_not_verified | 6 | Useful candidate without a verified calibrated lookup in this audit |

## Useful first set

Grid spacings below are service postings near the equator unless stated;
they are not guarantees of effective instrument resolution. Longitude distances
shrink with latitude. Coverage must come from the actual product footprint.

| Body | Variables | Verified access / support | Use |
| --- | --- | --- | --- |
| Both | Elevation; derive mean/min/max elevation, slope mean/p90/max, relief, aspect components, topographic position, roughness and fraction below a chosen slope | Trek numeric DEM samples. Moon global LOLA v06 ~118 m; Mars blended HRSC/MOLA ~200 m, MOLA ~463 m. Selected local DEMs can be finer | Primary terrain analogue and traversability screening |
| Moon | Noon, midnight, minimum and maximum bolometric surface temperatures; derive max-minus-min | Diviner numeric samples; tested global service grid 0.5° (~15.2 km), compiled 2009–2015 observations | Thermal setting, with surface-temperature and time-definition alignment |
| Moon | Long-term sunlight visibility fraction; Earth visibility fraction | Polar numeric products with 240 m and 60 m postings | Energy/communications context; modeled visibility rather than instantaneous power or a radio-link guarantee |
| Moon | Permanently shadowed-region intersection / covered fraction; distance to mapped surface-ice detection | Polar polygon/point spatial queries | Cold traps and mapped resource evidence |
| Both | Geologic unit, description and mapped interpretation | USGS map polygons via Trek | Categorical geological context; body-specific legends need an explicit cross-body mapping |
| Mars | Modeled annual approximate mean, minimum and maximum surface temperatures; derive range | ASU KRC model cutouts, 8 ppd (~7.4 km) | Thermal setting; model outputs, not current weather |
| Mars | Lambert albedo | ASU TES, 8 ppd (~7.4 km) | Reflectance/heating and surface context |
| Mars | Apparent thermal inertia | ASU TES ~7.4 km; Putzig nighttime ~3 km; THEMIS nominal ~100 m | Surface thermal response/material context; retain model and quality metadata |
| Mars | Surface dust-cover index | ASU TES 16 ppd (~3.7 km) | Surface dust context; not atmospheric dust opacity |
| Mars | Water-equivalent hydrogen | ASU Odyssey neutron-derived map, 2 ppd (~29.6 km) | Broad hydrogen/resource proxy, with measurement footprint potentially larger than grid spacing |
| Mars | Hydrous-mineral detections/classes, chloride-bearing detections, dune fields, valley networks, mapped glacial context | Trek spatial feature queries | Presence, category or distance to mapped features; empty result is not proof of absence |

For the Mars fans/glaciers/gullies service, this audit checked only its first
non-group layer, glacial valleys. It does not verify every sublayer in that
service. Feature support and mapping completeness must be inspected before
turning detections into an area fraction or an absence claim.

### Additional composition and material variables

**Moon:** Kaguya olivine, clinopyroxene, orthopyroxene, plagioclase, FeO, optical
maturity, modeled plagioclase grain size and mineral fit criterion all return
numeric values. The tested numeric services are **0.25° (~7.6 km)** and cover
50°S–50°N; their imagery is advertised at much finer spacing. The four mineral
channels sum to approximately one at all three controls, although source
documentation describes weight percentages. A ×100 display conversion is a
plausible inference, not a silently applied verified scale. FeO already returns
percent-point values. Submicroscopic metallic iron scale remains unresolved.
The [authors' source description](https://www.hou.usra.edu/meetings/lpsc2016/pdf/2994.pdf)
supports the mineral definitions and scientific abundance convention.

Other numeric lunar candidates are thorium (ppm), elemental titanium (wt% Ti,
not TiO2), Christiansen-feature wavelength, radar circular-polarization ratio
and backscatter. Regional NAC Site B supplies **detected rock counts per
100×100 m cell**, rather than rock area fraction. Site C encoding and regional
NAC roughness semantics remain unresolved. Coarse gamma-ray products are broad
composition context, not local resource concentrations or radiation-dose maps.

**Mars:** verified TES spectral-model channels include olivine, clinopyroxene,
plagioclase, K-feldspar, carbonate, hematite, amphibole, quartz, sulfate and
surface dust, plus model RMS error. These are coarse **areal model fractions**
at 4 ppd (~14.8 km), not elemental weight percentages. Keep the fit-error channel
and source NODATA with the abundances. See [TES products](https://tes.mars.asu.edu/products/)
and the [ASU numeric-map documentation](https://jmars.mars.asu.edu/numeric-maps).

### Useful but conditional or still unresolved

- Lunar polar water-equivalent hydrogen: northern numeric samples work at 5 km
  postings, while the southern service timed out. Published source-family maps
  use wt% WEH, but the exact Trek product lacks sufficient lineage/scaling
  metadata. Do not multiply raw values by 100 or label them calibrated percentages
  until that is resolved.
- Lunar ice-stability depth: northern numeric samples work at 240 m postings.
  The [model paper](https://www.hou.usra.edu/meetings/lunarvolatiles2018/pdf/5038.pdf)
  supports depths within the upper 2.5 m. The exact Trek variant and its capped
  2.5 values still require confirmation. This predicts where ice could survive,
  not where ice was detected or the depth to a deposit.
- KPLO potassium, uranium and thermal-neutron channels return numbers, but exact
  current raster scales remain unconfirmed. A [2024 team paper](https://www.hou.usra.edu/meetings/lpsc2024/pdf/1884.pdf)
  describes preliminary elemental counting maps; it cannot establish that the
  currently hosted raster is calibrated to wt% or ppm. The thermal-neutron
  channel is a detector-derived proxy, not temperature.
- Global lunar albedo, Diviner thermal-property/rock-area-fraction products,
  Mars calibrated rock abundance and regional [SWIM ice evidence](https://swim.psi.edu/)
  are useful next sources, but this audit did not verify calibrated small-cutout
  access for them. NAC rock counts cannot substitute for rock area fraction.
- Atmospheric pressure, air temperature, wind and atmospheric dust on Mars need
  an explicitly seasonal/time-dependent source or model. No static Trek lookup
  was verified here. Lunar air-temperature matching is not meaningful.
- Gravity, geoid, crustal thickness/density and magnetic models are numeric
  candidates recorded in the catalog, with product semantics still to pin down.
  They have lower priority for surface Earth-analogue matching. Crustal grain
  density is not surface soil bulk density.

## Findings that change how we should sample

1. **Reject the published Mars Celsius temperature-range layer.** In all 64
   checked pixels it equals `(maximum - minimum) - 273.15`, to float rounding,
   giving negative ranges. Derive `maximum - minimum` from matched valid source
   pixels. The evidence is in [validation.json](validation.json); this is a
   finding about the tested cutout/version, not a claim about every product.
2. **A 4×4 km request does not create 4 km source detail.** Terrain can support
   local statistics; many thermal, composition and hydrogen products cannot.
   Return source support/resolution and missingness alongside their values.
3. **Do not derive a 4 km slope from an arbitrary 3×3 grid.** Fetch appropriate
   native DEM support and a halo; use body-specific distances, a defined gradient
   baseline, area weighting and valid-pixel support. Roughness also needs a
   mathematical definition and baseline. The audit's point probes do not verify
   those area computations.
4. **Align Earth and planetary definitions.** Compare surface temperature with
   compatible Earth surface temperature, not air temperature; align seasons,
   local-time definition, footprint and summaries where possible. Different
   vertical datums make raw elevation a context field rather than a universally
   comparable height. Geological category codes and abundance methods also differ.
5. **Missing is not zero.** Respect source-specific NODATA, masks and mapping
   completeness. Example ASU sentinels include -32768 (THEMIS), -1 (TES mineral
   channels) and ~0.85000002 (TES dust index). Preserve zero when valid. Negative
   Christiansen wavelengths in source statistics show why finite-number checks
   alone are insufficient. NDVI/EVI on Moon/Mars should be unavailable/not
   applicable rather than fabricated zeroes.

## Access and backend recommendation

Use a backend sampler with provider adapters and a cache. NASA Trek
`ImageServer/getSamples` returns numeric point data; ArcGIS image export or
another verified numeric cutout is needed for native-grid window statistics.
The vector services accept spatial intersection queries. The
[ArcGIS getSamples documentation](https://developers.arcgis.com/rest/services-reference/enterprise/get-samples/)
describes request semantics; planetary native spatial references must be preserved.

ASU's endpoint `https://ms-mars.mars.asu.edu/alb32` accepts WMS 1.1.1 `GetMap`
with `SRS=JMARS:4` and `FORMAT=image/vicar`. The audit decoded tiny REAL, DOUB,
HALF and FULL single-band VICAR responses. A REAL 8×8 response was 608 bytes.
Keep the format slash literal: this endpoint rejected the percent-encoded MIME
slash in our requests. HTTP 200 can contain an XML exception, so check the body.
An asymmetric MOLA control supports east-positive longitude for the tested
request ([orientation evidence](mars-orientation-check.json)); complete CRS,
latitude convention, wraparound and polar behavior still need adapter tests.

Record each value's source/version, unit and normalization, temporal support,
native grid spacing, effective resolution if known, requested footprint, valid
support fraction, quality flags, missing reason and retrieval time. No user
login was needed for the tested calls. Availability, production latency,
provider usage terms and reliability were not load-tested.

Suggested order: terrain and thermal profile first; then Mars albedo, thermal
inertia and dust, lunar polar illumination/shadow, and mapped geology/detections;
then composition and resource proxies after their normalization/quality checks.
Only fields with a compatible Earth counterpart should participate in analogue
scoring; the others can remain target context.

## Evidence and reproduction

- [trek-audit.json](trek-audit.json): inventories, full selected-service metadata,
  request URLs, response hashes, timings, native-coordinate probes and responses.
- [mars-cutout-audit.json](mars-cutout-audit.json): bounded 8×8 requests, decoded
  headers/raw values and unsuccessful name probes.
- [mars-source-metadata.json](mars-source-metadata.json): selected factual
  capabilities keywords, grid density, units, NODATA and extent information.
- [source-descriptions.json](source-descriptions.json) and
  [lunar-unit-review.json](lunar-unit-review.json): source URLs, interpretations
  and distinctions between exact-product evidence and source-family inference.
- [validation.json](validation.json): artifact checks and observed inconsistencies.

```sh
# Network refresh: bounded requests, four/three workers, no global rasters.
python scripts/inspect_planetary_sources.py --retry
python scripts/inspect_mars_numeric_sources.py

# Regenerate the catalog and run offline evidence/decoder checks.
python scripts/build_planetary_catalog.py
python scripts/validate_planetary_sources.py
python scripts/test_planetary_sources.py
```

The metadata and orientation evidence are separate reviewed snapshots; the two
refresh commands do not replace them. The five offline decoder/classifier tests
and saved-evidence validation passed. These checks establish parsing and observed
consistency, not global scientific validation or a production sampler.

Additional primary references: [lunar polar illumination models](https://pgda.gsfc.nasa.gov/products/69),
[Diviner PDS composite product](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LRO-L-DLRE-5-GCP-V1.0),
[Lunar Prospector reduced products](https://pds-geosciences.wustl.edu/missions/lunarp/reduced_special.html),
[THEMIS thermal-inertia methods and units](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JE002735).

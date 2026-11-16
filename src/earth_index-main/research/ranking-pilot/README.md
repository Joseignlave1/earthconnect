# Generous retrieval and continuous ranking experiment

This evaluates whether the **existing data** supports broad candidate retrieval
and preference-dependent ranking. It uses the 27-patch raster bundle from the
[environmental stress pilot](../stress-pilot/README.md). No additional data,
indexes, tables, persistent materialized caches or schema changes are required.

The rules in [spec.json](spec.json) were fixed before the first ranking run.
The geographic sample was already known; this is an exploratory scoring
experiment, not a blind scientific benchmark or calibrated analog probability.
Site names, roles and expected environmental labels do not enter the SQL score.

## Measured result

The timings below are the original experiment. A subsequent
[SQL optimization](OPTIMIZATION.md) reduces query medians to **1.10–1.29 s**
without changing the data, scoring rules or complete result ordering. Use
`scripts/rank_pilot.py` for the optimized single-query path.

A separate [measurement-layout benchmark](VALUE_LAYOUT.md) isolates per-cell
scoring without sorting or window aggregation. Temporary decoded cell rows
outperform both raster access and quantized cell rows; no production table or
API has been switched to that representation.

The current data supports broad retrieval and preference-dependent ranking on
this selected sample. The shared gate retains **106,277 candidate pixels**,
excludes 130,182 pixels on observed conditions, and marks 33,541 as uncertain.
There are 270,000 geographic pixels in total. The gate preserves every pixel
in the best-scoring ungated 1% for each tested profile; this is only consistency
with the chosen score, not a claim of perfect scientific recall.

| Preference | Positive-score windows | Distinct patches with supported results | Median SQL time |
| --- | ---: | ---: | ---: |
| Hot / dry / bare, balanced | 1,320 | 18 | 8.11 s |
| Prioritize heat | 1,320 | 18 | 7.97 s |
| Prioritize flatness | 1,320 | 18 | 7.75 s |
| Cold / bare / frozen ground | 458 | 6 | 7.71 s |

Each window is 10 × 10 km, and each profile evaluates all 2,700 windows. A
positive result is not a claim of sufficient suitability: weak results remain
available much farther down the ranking. The scores are dimensionless research
preferences, not calibrated probabilities.

| Rank | Balanced hot/dry | Heat first | Flatness first | Cold/frozen ground |
| ---: | --- | --- | --- | --- |
| 1 | Central Sahara (91.1) | Central Australian desert (86.2) | Central Sahara (93.2) | Haughton Crater (83.0) |
| 2 | Central Australian desert (87.9) | Central Sahara (84.5) | Central Australian desert (91.3) | Axel Heiberg Island (72.4) |
| 3 | Death Valley / Badwater vicinity (86.1) | Death Valley / Badwater vicinity (84.2) | Gobi desert (90.0) | Central Svalbard (56.4) |
| 4 | Wadi Rum / southern Jordan (83.6) | Wadi Rum / southern Jordan (76.2) | Namib dune desert (89.9) | Everest / central Himalaya (49.0) |
| 5 | Gobi desert (83.5) | Gobi desert (72.4) | Death Valley / Badwater vicinity (89.3) | Central Iceland (28.7) |
| 6 | Namib dune desert (82.3) | Namib dune desert (71.0) | Wadi Rum / southern Jordan (88.3) | Central Alps (14.3) |
| 7 | Atacama / northern Chile (66.0) | Almeria / southeast Spain (47.9) | Atacama / northern Chile (83.9) | — |
| 8 | Almeria / southeast Spain (51.5) | Atacama / northern Chile (41.8) | Salar de Uyuni (72.7) | — |
| 9 | Salar de Uyuni (47.4) | Salar de Uyuni (21.9) | Haughton Crater (57.1) | — |
| 10 | Haughton Crater (31.4) | Haughton Crater (11.0) | Almeria / southeast Spain (56.8) | — |

This table shows one best measured window per sampled patch for geographic
comparison. All raw windows remain in the artifacts, with their actual ranking
unit and location. Compare scores within a preference column.

Concrete findings:

* Central Australia moves above Sahara when heat receives more weight. Its
  selected heat-first window averages about 33.19°C in the warmest monthly
  mean, versus 30.59°C in the selected Sahara window. Sahara is much drier,
  explaining why the balanced preference favors it. These are coarse-source
  climate values, not independent local 1 km weather observations.
* Low-slope Namib and Gobi windows move up when flatness receives more weight.
  Uyuni remains available, with a higher flatness-oriented score despite its
  much cooler summers. Flatness-first retains other preferences; it does not
  mean sorting exclusively by slope.
* Svalbard is third among the supported cold-ground patches, behind Haughton
  and Axel Heiberg. The previous strict winter cutoff no longer removes it.
* Himalaya is fourth for cold ground and Iceland fifth. Those locations were
  control/borderline hypotheses in the earlier binary test. Their selected
  windows can share some requested conditions, so their appearance is not
  automatically a false positive. Local geology and fine terrain remain untested.
* Tokyo, Uruguay and Mekong retain a few eligible pixels but their best whole-
  window scores remain very low. The broad gate allows marginal places to be
  returned while ranking places with much larger supported footprints higher.
* Every tested single-weight perturbation of ±20% retains the same five leading
  patches for that preference. This is top-five **membership** stability; it
  does not establish an invariant internal order or scientifically optimal weights.

## Missingness, quality and geographic repetition

Antarctica has no supported cold-ground result, but its best optimistic bound
is about 97.4/100 versus a supported Haughton score of 83.0/100. This means
missing measurements could hide a competitive candidate; 97.4 is a theoretical
upper bound, not an estimate of its actual suitability. It belongs in a data-
completion queue, not the normal measured ranking.

The Pacific also gets high optimistic bounds for some hot/flat preferences
because missing land-cover evidence does not establish land, barrenness or
usable ground. It does not enter the supported ranking. This exposes a real
limitation of the current variables: a reliable land/water mask would distinguish
known ocean from missing land evidence. An uncertainty queue alone is not a
qualified list of promising sites.

An independent [coverage audit](../pilots/database/ranking-coverage-audit.json)
requires 90% and 99% usable source support instead of the permissive baseline.
The leading five patches remain the same for balanced heat, heat-first and
cold-ground preferences. For flatness-first, Wadi Rum replaces Death Valley
in the top five. Most leading rankings therefore survive substantially stricter
coverage requirements, with one meaningful near-top swap. The audit is an
offline sensitivity calculation, not a new SQL benchmark or changed default.

Component evidence measures the fraction of scoring weights with usable inputs.
It is not statistical confidence, measurement accuracy, or a guarantee of common
valid support below a pixel. Native source-valid fractions are recorded in the
coverage audit; coarse climate and mean-slope limitations remain even at full
coverage.

The raw top 20 windows come from only one or two patches, depending on the
preference. Neighboring strong windows can dominate a naive results list. We
therefore report both raw results and the explicit one-window-per-patch
diagnostic. Useful geographic grouping/diversity remains a result presentation
choice; it does not require new source variables or an index change.

## Correctness and cost

All 10,800 SQL windows across four preferences match independently calculated
NumPy scores, bounds and counts. Maximum score/evidence disagreement is below
3×10⁻¹⁵. Five targeted property checks also pass, including SQL fixtures proving
physical zero stays distinct from NULL and that coincident hot/flat pixels beat
disjoint variable extrema.

Twelve measured SQL executions (three per profile) complete within the existing
10-second statement limit. Median end-to-end times are about 7.7–8.1 seconds,
including connection setup, scanning the current raster view, scoring and
returning all 2,700 windows. The complete evaluation, verification and weight-
sensitivity run takes about 110 seconds. This is sufficient for the requested
measurement; it is not an instant interactive or globally scalable query claim.
No new indexing was needed to establish whether ranking is useful.

All 15 existing project index definitions have identical before/after
fingerprints. Source rasters, existing bundles and migrations are unchanged.

See the [full ranking report](../pilots/database/ranking-evaluation.json),
[retained SQL and window artifacts](../pilots/database/ranking-artifacts/), and
[coverage sensitivity report](../pilots/database/ranking-coverage-audit.json).

## Retrieval and score definition

The shared broad gate retains pixels with at least 5% classified bare ground,
mean slope no greater than 35°, at least 20% accepted land-cover support and at
least 50% terrain support. These are permissive gates for this **bare-land
search**, not universal requirements for every analog task. There are no hard
temperature, precipitation or permafrost cutoffs.

Observed contradictions exclude a pixel. Missing or inadequately supported
gate measurements make its eligibility unknown. Unknowns are kept in the
evaluation and review queue rather than called environmentally unsuitable.

Four profiles use smooth preferences:

* Balanced hot/dry/bare terrain.
* Greater emphasis on summer heat.
* Greater emphasis on low mean slope.
* Cold summers/winters and modeled permafrost.

Heat/cold use logistic curves; rain, mean slope and coarse neighborhood relief
use exponential preferences; bare ground and permafrost use their fractions.
The versioned physical scales and weights are explicit in the specification.
The hot profiles also use a small relief component, which inherits the
EarthEnv elevation metadata caveat. No mineral, basalt or crater classifier is
implied by these profiles.

Component values are combined with a weighted geometric mean **at the same
pixel**. These pixel scores are then averaged over fixed, nonoverlapping
10 × 10 km windows. An isolated good pixel cannot give an entire window its
maximum score, and independent hot/flat maxima from different locations are
never combined. This does not resolve variation below the source support or
certify fine-scale trafficability. Fixed windows can split useful footprints;
they are a reproducible ranking unit rather than an optimal site boundary.

## Missing evidence

Each unknown component has possible score range [0,1]. The geometric score is
reported with conservative lower and optimistic upper bounds. Because an
unknown factor could be zero, a pixel with any missing weighted component has
a zero lower bound; this is an uncertainty bound, not an imputed observation.
Missing gate evidence similarly leaves eligibility unresolved.

The primary ranking uses supported lower-bound window scores. Evidence coverage
and upper bounds remain visible, with a separate queue for potentially useful
but incompletely measured windows. Scores are never renormalized over only
observed dimensions, which could otherwise reward missing information. These
bounds describe missing measurements only; they are not statistical confidence
intervals for source errors or geological suitability.

The same curves can be used to rank arbitrary candidate subsets in SQL. The
experiment reports both raw top windows and a geographic diagnostic containing
one best window per sampled patch. Geographic grouping is explicit; it does not
change the raw ranking or suppress neighboring results silently.

## Validation and diagnostics

The evaluator independently computes every window's candidate/unknown/excluded
counts, score bounds and evidence in NumPy, then compares them with actual SQL
results. The SQL uses the existing read-only research view and a 10-second
statement timeout. Three samples per profile measure local query cost.

It also measures:

* Whether the broad gate preserves the scoring model's best ungated pixels.
  This is a self-consistency check, **not independent scientific recall**.
* Which sampled environments appear near the top and how far down weak matches
  fall. Existing environmental labels are hypotheses, not relevance judgments.
* Changes between heat-first, flatness-first and cold-ground preferences.
* Top-five patch stability when each individual weight changes by ±20%, with
  weights renormalized.
* Geographic duplication among the raw top windows, and promising upper-bound
  candidates with incomplete evidence.

Database index definitions are fingerprinted before and after the experiment.
The specification and source bundle hashes are recorded. SQL arithmetic checks
do not remove the underlying scientific limitations: coarse climate/lithology,
mean-slope aggregation, mixed dates, source metadata conflicts and incomplete
polar coverage remain material.

## Reproduce

First load the existing stress raster bundle as described in its README, then:

```sh
docker compose run --rm --entrypoint python loader scripts/evaluate_ranking_pilot.py --artifacts-dir /reports/ranking-artifacts --report /reports/ranking-evaluation.json
```

The Compose workspace is read-only; generated reports, SQL and window artifacts
go through the writable `/reports` mount. On the host they appear under
`research/pilots/database/`. The retained SQL files use a bundle parameter and
the existing research view, so they can be inspected or reused by the Rust API.
Scores are research preference values, not percentages of proven suitability.

Run the focused semantic checks and optional coverage audit with:

```sh
docker compose run --rm --entrypoint python loader scripts/test_ranking_pilot.py
docker compose run --rm --entrypoint python loader scripts/audit_ranking_coverage.py --ranking-report /reports/ranking-evaluation.json --report /reports/ranking-coverage-audit.json
```

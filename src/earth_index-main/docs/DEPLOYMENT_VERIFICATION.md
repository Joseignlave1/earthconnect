# Initial Fly deployment verification

Verified 2026-11-15 against <https://earth-index-crisrpia.fly.dev>.

- Source release: `8b6438cca56b2cb210b161e29a8ba41afa64bd27` on private GitHub
  repository `CrisRPia/earth_index`.
- Application image SHA-256:
  `33784453b76a56c5369074c2160bc0cd3a4022404036cf9bf2cde962d579c7f2`.
- Serving database: 762,589,184 bytes. Uploaded file checksum matches the local
  snapshot; see [deployment operations](DEPLOYMENT.md).
- One Machine, `80d45da66044e8`, in `iad`: four performance vCPUs, 8192 MiB RAM,
  10 GB encrypted persistent volume. No spare Machine or remote build Machine.
- Configuration inspected on Fly: automatic stop/start enabled, minimum running
  Machines zero, restart on failure, four DuckDB threads and 5 GB engine limit.

## Checks

- Local Rust suite: 26 passed; database-dependent integration tests are opt-in.
- Local frontend: 34 unit tests and six component tests passed; Svelte/TypeScript
  check has zero errors and warnings. Five optional integration tests were then
  explicitly run against Fly and all passed.
- Public HTTPS serves the demo and built JavaScript assets. HTTP/2 native gRPC
  Health returned ready with gRPC status zero. Binary gRPC-Web Health, Metadata,
  repeated named SQL parameters, SQL NULL and packed uint16 edge values passed.
- Full-world query from the real query builder returned all 460 region arrays,
  71,101,280 score bytes, with the same combined SHA-256 as the local deployment:
  `adf3b8305bb4da28ab81a498fa64d4c309622595ddff0be1de9905d8e5849cdc`.
  This release probe uses slope-only scoring to verify transport/data identity;
  it does not change the demo's multi-variable defaults or assess scientific
  matching quality. Positive scores: 28,393,314; missing scores: 6,486,660.
- Planetary sampling at Moon (0, 0) returned 36 available measurements out of 59;
  Mars (23.47, 0.67) returned 30 out of 36. Both responses were non-partial for
  these probes. Upstream services can have different coverage and availability
  at other coordinates.
- Headless Chromium loaded all three maps, selected a lunar point, populated ten
  scoring criteria (five measured terrain, four approximate thermal, one
  preference), ran the full query and rendered scores with no uncaught JavaScript
  errors. The selected point had some unavailable upstream sources, correctly
  reported as partial. The result included all 460 regions.
- Stopped the Machine, requested its public HTTPS health endpoint, and observed
  Fly Proxy start the same Machine and the API become ready. This tests wake-up
  after a stop; automatic idle stopping is configured rather than timed here.

Some local DNS lookups briefly returned only IPv6 during initial propagation;
this acquisition computer has no IPv6 route. Fixed-public-IPv4 checks preserved
the HTTPS hostname and normal certificate validation. Public DNS-over-HTTPS
confirmed the application's IPv4 record; a full-world gRPC-Web check also passed
using normal hostname resolution. No application's DNS overrides are deployed.

The Machine is deliberately left stopped after verification. Opening the demo
wakes it. This report is functional release evidence, not a Fly throughput or
latency benchmark.

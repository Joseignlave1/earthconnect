# Fly.io demo

Demo: <https://earth-index-crisrpia.fly.dev/demo/>

See [initial deployment verification](DEPLOYMENT_VERIFICATION.md) for the tested
image/database identity and live checks.

The [Fly configuration](../fly.toml) runs one Linux amd64 API/demo container in
Ashburn (`iad`), with 4 performance vCPUs, 8 GiB RAM and a 10 GB encrypted volume.
DuckDB uses four threads and a 5 GB memory limit, leaving room for the Rust
process, query response buffers and the operating system. PostgreSQL and grpcui
are not deployed. Native gRPC, gRPC-Web and the frontend share HTTPS port 443;
Fly forwards HTTP/2 cleartext to Axum on port 8080. The hosted `/` and `/docs`
redirect to `/demo/`; reflection and `/proto/earth_index.proto` remain available
for external API tools. Local Compose still includes grpcui.

## Scale to zero

`auto_stop_machines = "stop"`, `auto_start_machines = true` and
`min_machines_running = 0` stop the Machine after an idle period and wake it for
the next request. Compute is billed while it runs, including the idle period;
the volume and stopped root filesystem remain billed while stopped. Avoid
external uptime monitors that continuously request the app, since they can
prevent it becoming idle. Cold starts also discard the in-memory NASA sample
cache and database caches. Fly health checks do not replace end-to-end checks.

Only one Machine is provisioned. Use `--ha=false` on deploys to avoid creating a
spare. The local immutable dataset is the recovery copy; preserve it outside the
Fly volume as well. Volume snapshots are enabled, but this demo accepts downtime
on host failure and during deployments.

## Deploy application changes

Install/login to `flyctl`. If installed outside PATH:

```sh
export PATH="$HOME/.fly/bin:$PATH"
fly auth login
fly config validate
fly deploy --local-only --ha=false
```

The existing Dockerfile builds the demo and Rust API. The database, `.env`, NASA
credentials, raw rasters and checkpoints are excluded from the image and Git.
The first release reused the locally tested image without recompiling it:

```sh
fly auth docker
docker tag earth-index-api registry.fly.io/earth-index-crisrpia:release-20261115
docker push registry.fly.io/earth-index-crisrpia:release-20261115
fly deploy --image registry.fly.io/earth-index-crisrpia:release-20261115 --ha=false
```

## Database bootstrap and replacement

The serving database is read-only inside the application. Files uploaded over
the private administration connection are separate deployment artifacts. The
initial snapshot is 762,589,184 bytes, SHA-256:

```text
f6d52856103f557a514ff01c0525798f6e9a8e1ec9bd132421383a6cb81eaf4a
```

For a new app, create its volume and a temporary sleeping Machine using the API
image, before the initial application deploy. The metadata lets `fly deploy`
adopt this same Machine after the upload:

```sh
fly apps create earth-index-crisrpia --org personal
fly ips allocate-v4 --shared
fly ips allocate-v6
fly volumes create earth_data --region iad --size 10 --vm-size performance-4x --vm-memory 8192
fly machine run registry.fly.io/earth-index-crisrpia:release-20261115 infinity \
  --region iad --volume earth_data:/data --entrypoint /bin/sleep \
  --vm-size performance-4x --vm-memory 8192 \
  --metadata fly_process_group=app --metadata fly_platform_version=v2 --restart no
```

For the initial upload or a later replacement, select the application's Machine
ID with `fly machine list`. Start it first if it is stopped. Upload to a new
temporary filename, compare the local and remote checksum, then rename atomically
and restart the application. Do not overwrite the file while DuckDB has it open.

```sh
sha256sum db/runtime/earth.duckdb
fly ssh sftp put db/runtime/earth.duckdb /data/earth.duckdb.upload --mode 0444
fly ssh console -C 'sha256sum /data/earth.duckdb.upload'
# Continue only when the checksums match:
fly ssh console -C 'mv /data/earth.duckdb.upload /data/earth.duckdb'
# Initial bootstrap: deploy the application. Later replacements: restart it.
fly deploy --image registry.fly.io/earth-index-crisrpia:release-20261115 --ha=false
```

If the private SSH connection fails over UDP, `fly wireguard websockets enable`
followed by `fly agent restart` enables Fly's WebSocket tunnel fallback. An
interrupted upload leaves the previous serving file intact; remove only the
incomplete `.upload` file before retrying. Never deploy NASA ingestion tokens;
the serving API uses public planetary sources.

## Verify and operate

```sh
curl --fail https://earth-index-crisrpia.fly.dev/healthz
MATCHING_API=https://earth-index-crisrpia.fly.dev npm run verify-deployment --prefix demo
MATCHING_API=https://earth-index-crisrpia.fly.dev npm run verify-deployment --prefix demo -- --world
fly status
fly checks list
fly logs
```

The lightweight check verifies demo assets, gRPC-Web health/metadata, named SQL
parameters, the global dataset and packed scores. `--world` exercises the real
query builder and validates every returned region's packed score shape and codes.
The protobuf contract is exposed through reflection for native gRPC clients.

This is an unauthenticated public demo of a read-only SQL driver. Arbitrary SQL
and the explicitly requested unlimited query duration remain available. The
engine restricts external file/network access, uses read-only transactions and
limits active database queries to two; this is not an authenticated multi-tenant
service. Source sampling has independent finite budgets. For controlled demos,
stop the Machine manually with `fly machine stop <id>`; the next public request
will wake it again. To prevent waking, change the public service configuration
rather than relying on a manual stop.

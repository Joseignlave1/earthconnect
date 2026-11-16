"""Run/checkpoint the global 25-band import. Safe to stop and restart.

No database writes or publication. Completed, verified U16 tiles are retained.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import signal
import time

from raster_source_cache import atomic_json
from world_features import process_tile, sha_file, valid_checkpoint, ROOT
import world_grid as grid
from world_sources import thermal_catalog, cover_catalog, shared_sources, tile_sources, log

STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    log('Stop requested; finishing in-flight chunks. Completed checkpoints are safe.')


def dependency_hashes():
    # Frozen in the Docker image; bind-mounted output cannot change processor code.
    names = ['world_import.py', 'world_grid.py', 'world_features.py', 'world_sources.py',
             'expansion_features.py', 'thermal_import.py', 'thermal_pilot.py',
             'priority_pilot.py', 'raster_source_cache.py', 'raster_overlap.py',
             'prepare_nasa_subsets.py', 'process_native_temperature.py', 'process_nasa_temperature.py']
    return {name: sha_file(ROOT / 'scripts' / name) for name in names}


def run(args):
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'research'):
        raise ValueError('Output must be under research/')
    for child in ('sources', 'catalog', 'checkpoints', 'tiles', 'raw-cache', 'shared'):
        (out / child).mkdir(parents=True, exist_ok=True)
    methods = ROOT / 'db/WORLD_IMPORT.md'
    if not (out / 'METHODS.md').exists():
        shutil.copyfile(methods, out / 'METHODS.md')
    lock = (out / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another global importer owns this output') from None
    started = datetime.now(timezone.utc).isoformat()
    status = dict(status='preparing', session_started_at=started, workers=args.workers,
                  download_workers=8, output=str(out), grid=grid.definition())

    def report(**updates):
        status.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'status.json', status)

    report(stage='catalog')
    log('Global import starting/resuming; each native tile is an independent checkpoint')
    catalog = thermal_catalog(out)
    landcover = cover_catalog(out)
    plan = dict(grid=grid.definition(), tiles=grid.tiles(), thermal_year=2023, landcover_year=2024,
                quantization_overrides=dict(annual_precip_mm=dict(scale=1., offset=-1.)),
                power_baseline='2001-2020 public bulk climatology; pilot API used 2001-2025',
                thermal_catalog_hashes={p.name: sha_file(p) for p in sorted((out / 'catalog').glob('2023-*.json'))},
                landcover_catalog_sha256=sha_file(out / 'catalog' / 'landcover.json'),
                methods_sha256=sha_file(methods),
                dependencies=dependency_hashes(), spec_sha256=sha_file(ROOT / 'research/thermal-pilot/manifest.json'))
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    frozen = out / 'plan.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != plan:
            raise RuntimeError('Processor, grid, catalog or specification changed. Resume with the original image.')
    else:
        atomic_json(frozen, plan)
    keys = plan['tiles']
    if args.only:
        if set(args.only) - set(keys):
            raise ValueError('Unknown tile selection')
        keys = args.only
    report(stage='shared_sources', total_tiles=len(keys), run_identity=identity)
    shared_sources(out)
    done = {}
    for key in keys:
        previous = valid_checkpoint(out, key, identity)
        if previous:
            done[key] = previous
    queue = [key for key in keys if key not in done]
    controls = ['h18v07', 'h12v11', 'h08v05', 'h15v01']
    queue.sort(key=lambda key: (controls.index(key) if key in controls else len(controls), key not in catalog, key))
    failures = {}
    retries = {}
    active = {}
    preparing = {}
    ready = []
    report(status='running', stage='tiles', completed_tiles=len(done), remaining_tiles=len(queue))
    log(f'Resume verified {len(done)}/{len(keys)} complete tiles; {len(queue)} remain')
    workers = args.workers
    with ThreadPoolExecutor(max_workers=2) as transfers, ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as processors:
        while queue or preparing or ready or active:
            paused = (out / 'PAUSE').exists()
            if STOP:
                queue.clear()
                ready.clear()
            if not STOP and not paused:
                while ready and len(active) < workers:
                    key = ready.pop(0)
                    active[processors.submit(process_tile, (key, str(out), identity))] = key
                # Bounded lookahead: <=2 transfers and <=workers ready CPU jobs.
                while queue and len(preparing) < 2 and len(ready) < workers:
                    key = queue.pop(0)
                    if shutil.disk_usage(out).free < args.reserve_gb * 1e9:
                        (out / 'PAUSE').touch()
                        queue.insert(0, key)
                        log('Paused before exhausting disk reserve; completed chunks retained')
                        break
                    future = transfers.submit(tile_sources, key, catalog.get(key, []), landcover.get(key),
                                              out, args.token_file, 4)
                    preparing[future] = key
            futures = set(active) | set(preparing)
            finished = wait(futures, timeout=5, return_when=FIRST_COMPLETED).done if futures else set()
            if not futures:
                time.sleep(2)
            for future in finished:
                is_cpu = future in active
                key = (active if is_cpu else preparing).pop(future)
                try:
                    result = future.result()
                    if is_cpu:
                        done[key] = result
                        failures.pop(key, None)
                        # Source hashes remain in sources/<tile>.json after cache deletion.
                        shutil.rmtree(out / 'raw-cache' / key, ignore_errors=True)
                        log(f'COMPLETE {len(done)}/{len(keys)} {key}: 25 bands; '
                            f'{result["domain_cells"]} domain cells; {result["seconds"]:.1f}s processing')
                    elif not STOP:
                        ready.append(key)
                        total = sum(r['bytes'] for r in result['thermal']) + (result['landcover'] or {}).get('bytes', 0)
                        log(f'Downloads verified {key}: {total/1e6:.1f} MB; queued for processing')
                except Exception as error:
                    failures[key] = dict(type=type(error).__name__, message=str(error)[:300])
                    retries[key] = retries.get(key, 0) + 1
                    log(f'RETRY {key}: {type(error).__name__}; attempt {retries[key]} (details in status.json)')
                    if not STOP and retries[key] < 6:
                        queue.append(key)
            report(status='stopping' if STOP else 'paused' if paused else 'running',
                   completed_tiles=len(done), total_tiles=len(keys), active_tiles=list(active.values()),
                   downloading_tiles=list(preparing.values()), ready_tiles=ready.copy(),
                   remaining_tiles=len(keys) - len(done), errors=failures,
                   disk_free_gb=round(shutil.disk_usage(out).free / 1e9, 2))
            if STOP and not active and not preparing:
                break
    if len(done) == len(keys) and not args.only:
        layers = [layer for key in keys for layer in done[key]['layers']]
        # No global publication until all tile checkpoints validate once more.
        if not all(valid_checkpoint(out, key, identity) for key in keys):
            raise RuntimeError('Final checkpoint verification failed')
        manifest = dict(status='processed_pending_database_load', run_identity=identity,
                        generated_at=datetime.now(timezone.utc).isoformat(), grid=plan['grid'],
                        dependency_files=plan['dependencies'], tiles=keys, layers=layers,
                        total_domain_cells=sum(r['domain_cells'] for r in done.values()),
                        source_rasters_retained='Shared sources only; per-tile raw assets deleted after checkpoint.',
                        scientific_methods='METHODS.md', publication='Not loaded or published to PostgreSQL by this job.')
        atomic_json(out / 'manifest.json', manifest)
        report(status='completed', stage='ready_for_database_load', completed_tiles=len(done), errors={})
        log(f'GLOBAL COMPLETE: {len(keys)} tiles, {len(layers)} U16 layers; ready for database ingestion')
    elif STOP:
        report(status='stopped', stage='resume_with_same_image')
    elif args.only and len(done) == len(keys):
        report(status='subset_completed', stage='resume_without_only_for_global')
    else:
        report(status='failed', stage='retry_preserves_completed_tiles')
        raise RuntimeError('Some global chunks remain incomplete; restart to retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/world-4km')
    parser.add_argument('--token-file', type=Path, default=Path('/run/secrets/earthdata_token'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reserve-gb', type=float, default=20)
    parser.add_argument('--only', nargs='+', help='Validation subset; checkpoints remain reusable for global run')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Worker count must be positive')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        run(args)
    except Exception as error:
        path = args.output / 'status.json'
        if path.exists():
            status = json.loads(path.read_text())
            status.update(status='failed', error_type=type(error).__name__, error=str(error)[:300],
                          updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(path, status)
        raise


if __name__ == '__main__':
    main()

"""Bounded, deduplicated MODIS downloads with reusable connections and hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import queue
import ssl
import threading
import time
from urllib.parse import urlparse
import uuid

from thermal_pilot import Signer


def asset_key(href):
    return hashlib.sha256(href.encode()).hexdigest()


def asset_path(cache, href):
    return Path(cache)/(asset_key(href)+'.tif')


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
    try:
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SourceCache:
    def __init__(self, directory, *, max_bytes=64*1024*1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.pools = {}
        self.signer = Signer()
        self.signed_at = time.monotonic()

    def _sign(self, href):
        with self.lock:
            if time.monotonic()-self.signed_at > 1800:
                self.signer = Signer()
                self.signed_at = time.monotonic()
            return self.signer.sign(href)

    def _fetch(self, href):
        url = urlparse(self._sign(href))
        with self.lock:
            pool = self.pools.setdefault(url.hostname,queue.LifoQueue())
        start = time.perf_counter()
        for attempt in range(3):
            try: connection = pool.get_nowait()
            except queue.Empty:
                connection = http.client.HTTPSConnection(url.hostname,timeout=60,context=ssl.create_default_context())
            try:
                connection.request('GET',url.path+'?'+url.query,headers={
                    'User-Agent':'EarthIndex/tile-import','Accept-Encoding':'identity'})
                response = connection.getresponse()
                payload = response.read(self.max_bytes+1)
                if len(payload)>self.max_bytes:
                    connection.close()
                    raise RuntimeError('Source exceeds per-file download limit')
                if response.will_close: connection.close()
                else: pool.put(connection)
                if response.status in (429,500,502,503,504) and attempt<2:
                    delay=response.getheader('Retry-After','')
                    time.sleep(min(30,int(delay)) if delay.isdigit() else 2**attempt)
                    continue
                if response.status != 200:
                    raise RuntimeError(f'Source download failed (HTTP {response.status}); access signature omitted')
                if payload[:4] not in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'):
                    raise RuntimeError('Expected a TIFF source')
                return payload, dict(seconds=time.perf_counter()-start,retries=attempt)
            except (OSError,http.client.HTTPException):
                connection.close()
                if attempt==2:
                    raise RuntimeError('Source connection failed; access URL omitted') from None
        raise RuntimeError('Source download retries exhausted')

    def fetch(self, href):
        # Signer validates provider/HTTPS/unsigned URLs even on cache hits.
        self._sign(href)
        file=asset_path(self.directory,href); meta=file.with_suffix('.json')
        if file.exists() and meta.exists():
            try:
                previous=json.loads(meta.read_text())
                checksum=hashlib.sha256(file.read_bytes()).hexdigest()
                if previous['href']==href and previous['sha256']==checksum and previous['bytes']==file.stat().st_size:
                    return previous|dict(cached=True,seconds=0.,retries=0)
            except (ValueError,KeyError,OSError):
                pass
        payload,stats=self._fetch(href)
        result=dict(href=href,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        temporary=file.with_name(file.name+'.'+uuid.uuid4().hex+'.part')
        try:
            temporary.write_bytes(payload);temporary.replace(file);atomic_json(meta,result)
        finally:
            temporary.unlink(missing_ok=True)
        return result|dict(cached=False,**stats)

    def fetch_all(self, hrefs, workers=8):
        if workers<1: raise ValueError('Download workers must be positive')
        unique=sorted(set(hrefs))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.fetch,unique))
        finally:
            self.close()

    def close(self):
        for pool in self.pools.values():
            while not pool.empty():pool.get_nowait().close()

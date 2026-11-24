# Earth Connect asset CDN

Cloudflare Pages serves the release from the `earth-connect-assets` R2 bucket
through `https://earth-connect-assets.pages.dev/manifest.json`. The Pages
deployment contains the Worker and its R2 binding. Release uploads go directly
to the bucket and take effect without a Pages deployment.

Run these commands from this directory with Wrangler authenticated to the
account that owns the Pages project and bucket.

```sh
npx wrangler@4.129.0 pages deploy --branch main
```

Upload content-addressed assets and the versioned release manifest before
updating `manifest.json`. The Worker caches versioned paths at the edge and
requires clients to revalidate the current manifest. It forwards the stored
content type and exposes the R2 ETag to cross-origin clients.

The `/find` route and the default standalone HTML export use this manifest.
Its `assets` mapping and `scoreManifest` identify the logical runtime paths
and their content-addressed objects. The browser checks each newly downloaded
file's size and SHA-256 before saving it to IndexedDB. It revalidates the
manifest with `If-None-Match` and reuses files whose hashes carry forward to
the next release. When Web Locks is available, tabs share release
preparation and discard obsolete files after the replacement is fully saved.

Local tests and offline exports use the immutable release recorded in
[pinned-release.json](pinned-release.json). Run `npm run fetch:assets`
from the repository root to download the pinned binaries or verify existing
local copies. When updating the pin, keep the checkout's unversioned Earth
Score manifest in sync with the release's score manifest. The CDN version
uses content-addressed chunk filenames. The tracked manifest uses the original
filenames so offline tools can resolve local chunks.

"""Check public AppEEARS product/layer metadata without login or raster reads."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = [
    "MOD11A2.061",
    "MCD43A3.061",
    "MCD12Q1.061",
    "MOD09A1.061",
    "MOD13Q1.061",
    "MOD10A1.061",
    "MOD16A2GF.061",
    "MCD15A3H.061",
    "MOD17A2HGF.061",
    "MOD17A3HGF.061",
    "MCD64A1.061",
    "MCD19A2.061",
    "AG100.003",
    "SPL3SMP_E.006",
    "SPL3FTP_E.004",
    "SPL4SMGP.008",
    "M2T1NXAER.5.12.4",
]


def inspect(product):
    url = "https://appeears.earthdatacloud.nasa.gov/api/product/" + product
    result = {
        "product": product,
        "url": url,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with urlopen(url, timeout=45) as response:
            metadata = json.load(response)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Expected a nonempty layer catalog")
        result.update(
            available=True,
            layer_count=len(metadata),
            layers=list(metadata),
            metadata=metadata,
        )
    except HTTPError as e:
        result.update(available=False if e.code == 404 else None, status=e.code)
    except (OSError, ValueError) as e:
        result.update(available=None, error=type(e).__name__)
    return result


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, PRODUCTS))
    path = ROOT / "research/pilots/appeears-products.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(
            result["product"],
            result["available"],
            result.get("layer_count", result.get("status", result.get("error"))),
        )

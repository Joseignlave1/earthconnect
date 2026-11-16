"""Prepare bounded NASA temperature requests; optionally submit with private login.

Default invocation needs no credentials and does not submit remote jobs.
--submit --credentials-file PATH reads username:password from a private file.
Secrets and server responses containing account information are never written.
"""

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "pilots" / "nasa-subsets"
API = "https://appeears.earthdatacloud.nasa.gov/api"
LAYERS = [
    "LST_Day_1km",
    "LST_Night_1km",
    "QC_Day",
    "QC_Night",
    "Emis_31",
    "Emis_32",
    "Clear_sky_days",
    "Clear_sky_nights",
]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an authentication header to a redirect destination.
        return None


def request(path, *, data=None, authorization=None):
    headers = {"User-Agent": "earth-index-pilot/1"}
    if authorization:
        headers["Authorization"] = authorization
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(
        API + path,
        data=None if data is None else json.dumps(data).encode(),
        headers=headers,
    )
    try:
        with build_opener(NoRedirect).open(req, timeout=60) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except HTTPError as e:
        # Do not print response bodies, credentials, or authentication headers.
        raise RuntimeError(
            f"NASA API {path.split('/')[1]} returned HTTP {e.code}"
        ) from None
    except URLError:
        raise RuntimeError("NASA API connection failed") from None


def login_credentials(path):
    secret = path.read_text().strip()
    username, separator, password = secret.partition(":")
    if not separator or not username or not password:
        raise RuntimeError("Private login must contain username:password")
    login = request(
        "/login",
        data={},
        authorization="Basic " + base64.b64encode(secret.encode()).decode(),
    )
    return "Bearer " + login["token"]


def logout(bearer):
    try:
        request("/logout", data={}, authorization=bearer)
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    if args.submit and args.status:
        parser.error("choose --submit or --status")
    if (args.submit or args.status) and not args.credentials_file:
        parser.error("a private credentials file is required for NASA task operations")
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "research/pilots/appeears-products.json").read_text())
    product = next(p for p in catalog if p["product"] == "MOD11A2.061")
    if not all(layer in product["metadata"] for layer in LAYERS):
        raise RuntimeError(
            "Requested layers are absent from the checked product catalog"
        )
    inventory = json.loads((ROOT / "research/pilots/inventory.json").read_text())
    tasks = []
    for site in inventory["sites"]:
        payload = {
            "task_type": "area",
            "task_name": f"earth-index-{site['key']}-temperature-2024-v1",
            "params": {
                "dates": [{"startDate": "01-01-2024", "endDate": "12-31-2024"}],
                "layers": [
                    {"product": "MOD11A2.061", "layer": layer} for layer in LAYERS
                ],
                "geo": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"site": site["key"]},
                            "geometry": site["geometry"],
                        }
                    ],
                },
                "output": {"format": {"type": "geotiff"}, "projection": "native"},
            },
        }
        path = OUT / f"{site['key']}-temperature-2024.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        tasks.append(
            (site["key"], payload, hashlib.sha256(path.read_bytes()).hexdigest())
        )
    if not (args.submit or args.status):
        print(f"Prepared {len(tasks)} area requests; NASA login is required to submit.")
        return
    bearer = login_credentials(args.credentials_file)
    receipts_path = OUT / "receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
    try:
        for site, payload, digest in tasks:
            existing = next(
                (r for r in receipts if r["request_sha256"] == digest), None
            )
            if args.submit and existing:
                print(
                    f"{site}: existing task {existing['task_id']}; not submitted twice"
                )
                continue
            if args.status:
                if not existing:
                    print(f"{site}: not submitted")
                    continue
                result = request("/task/" + existing["task_id"], authorization=bearer)
                existing["status"] = result["status"]
                existing["checked_at"] = datetime.now(timezone.utc).isoformat()
                print(f"{site}: {existing['status']}")
            else:
                result = request("/task", data=payload, authorization=bearer)
                receipt = {
                    "site": site,
                    "task_id": result["task_id"],
                    "request_sha256": digest,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "status": "submitted",
                }
                receipts.append(receipt)
                print(f"{site}: submitted {receipt['task_id']}")
            receipts_path.write_text(json.dumps(receipts, indent=2) + "\n")
    finally:
        logout(bearer)


if __name__ == "__main__":
    main()

"""Generate local Docker database credentials; never print their values."""

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / ".env"
    if path.exists():
        print("Existing .env preserved.")
        return
    owner = secrets.token_hex(24)
    api = secrets.token_hex(24)
    contents = (
        "POSTGRES_PORT=5432\n"
        f"POSTGRES_PASSWORD={owner}\n"
        f"API_DB_PASSWORD={api}\n"
        f"DATABASE_URL=postgresql://earth_index_owner:{owner}@127.0.0.1:5432/earth_index\n"
        f"API_DATABASE_URL=postgresql://earth_index_api:{api}@127.0.0.1:5432/earth_index\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(contents)
    (ROOT / "research/pilots/database").mkdir(parents=True, exist_ok=True)
    print("Created private .env for Docker and host clients; values omitted.")


if __name__ == "__main__":
    main()

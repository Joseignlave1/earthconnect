"""Apply checksum-verified migrations and refresh API views; never import data."""
import os
import sys
from pathlib import Path

from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from load_pilots import migrate  # noqa: E402
from load_raster_pilot import connection  # noqa: E402


def main():
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(4182026, 1)")
        migrate(conn)
        if password := os.getenv("API_DB_PASSWORD"):
            conn.execute(sql.SQL("ALTER ROLE earth_index_api PASSWORD {}").format(sql.Literal(password)))
        conn.execute("SELECT catalog.refresh_api_datasources()")
    print("Database migrations and API datasource views are ready; no data imported.")


if __name__ == "__main__":
    main()

"""Download the Transfermarkt tables into data/raw/.

Usage:
    python -m src.download_data                 # all tables
    python -m src.download_data --tables transfers clubs
    python -m src.download_data --force         # download again

If the download fails, get the same CSV files from Kaggle
(https://www.kaggle.com/datasets/davidcariboo/player-scores) and put them
in data/raw/ (plain .csv or .csv.gz both work).
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from src.config import DATA_BASE_URL, RAW_DIR, TABLES


def table_url(table: str) -> str:
    return f"{DATA_BASE_URL}/{table}.csv.gz"


def find_local_file(table: str, raw_dir: Path = RAW_DIR) -> Path | None:
    """Return the local file of a table (.csv.gz or .csv), or None."""
    for suffix in (".csv.gz", ".csv"):
        path = raw_dir / f"{table}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def download_table(table: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_local_file(table, raw_dir)
    if existing and not force:
        print(f"[skip] {table}: already in {existing}")
        return existing

    target = raw_dir / f"{table}.csv.gz"
    tmp = target.with_suffix(".part")
    url = table_url(table)
    print(f"[get ] {table}: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "scoutai-data-pipeline"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.replace(target)
    print(f"[done] {table}: {target.stat().st_size / 1e6:.1f} MB")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", nargs="+", default=TABLES, choices=TABLES)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args(argv)

    failed = []
    for table in args.tables:
        try:
            download_table(table, force=args.force)
        except Exception as exc:  # network errors, HTTP errors
            print(f"[fail] {table}: {exc}", file=sys.stderr)
            failed.append(table)

    if failed:
        print(
            "\nSome tables could not be downloaded: " + ", ".join(failed)
            + "\nDownload them from Kaggle (davidcariboo/player-scores) into data/raw/.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the UCI Online Shoppers Purchasing Intention dataset (ID 468) and verify it.

Usage:  python scripts/download_data.py
Saves data/online_shoppers_intention.csv. Skips download if a verified copy already exists.
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "online_shoppers_intention.csv"
URL = ("https://archive.ics.uci.edu/static/public/468/"
       "online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip")
SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if DEST.exists():
        if sha256(DEST.read_bytes()) == SHA256:
            print(f"Verified existing file: {DEST}")
            return
        sys.exit(f"{DEST} exists but its SHA-256 does not match; remove it and rerun.")
    print(f"Downloading {URL}")
    with urlopen(Request(URL, headers={"User-Agent": "dataset-download/1.0"}), timeout=90) as r:
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if Path(n).name == DEST.name]
        if len(names) != 1:
            sys.exit(f"Expected one {DEST.name} in the archive, found {names}")
        data = archive.read(names[0])
    if sha256(data) != SHA256:
        sys.exit("Downloaded file SHA-256 does not match the expected value; not saved.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(data)
    print(f"Saved and verified: {DEST}")


if __name__ == "__main__":
    main()

"""Download MaleCNS v1.0 dataset from Google Cloud Storage."""

import hashlib
import json
from pathlib import Path
import urllib.request
import sys

DATASET_FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        "bytes": 14483314,
    },
    "body-neurotransmitters-male-cns-v1.0.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        "bytes": 43282834,
    },
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        "bytes": 1051241946,
    },
}


def download_dataset(target_dir: Path, include_edges: bool = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, info in DATASET_FILES.items():
        if filename.startswith("connectome-weights") and not include_edges:
            print(f"Skipping 1.05 GB edge weights file (run with --include-edges to download).")
            continue
        dest = target_dir / filename
        if dest.exists():
            print(f"{filename} already exists, checking checksum...")
        else:
            print(f"Downloading {filename} ({info['bytes'] / 1e6:.1f} MB)...")
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(info["url"], tmp)
            tmp.replace(dest)

        with open(dest, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != info["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {filename}!")
        print(f"Verified {filename} (SHA256: {digest[:12]}...)")


if __name__ == "__main__":
    include_edges = "--include-edges" in sys.argv
    root = Path(__file__).resolve().parents[1]
    download_dataset(root / "connectome_data" / "malecns_v1", include_edges=include_edges)

"""Download the public MWDET archive from its official GitHub repository."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/Yue-ZHAO/MWDET_Project/master/Data_Publish.zip"
SHA256 = "b12aa3f0b64e76c3fad5cbc3ad8c8ed469f97096121eb6b18397949d6b018e64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the public MWDET dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/mwdet"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "Data_Publish.zip"
    extracted = args.output_dir / "Data_Publish"
    if archive.exists() and not args.force:
        print(f"Using existing archive: {archive}")
    else:
        with urllib.request.urlopen(URL) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = sha256(archive)
    if actual != SHA256:
        raise ValueError(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    with zipfile.ZipFile(archive) as package:
        package.extractall(extracted)
    print(f"Verified and extracted MWDET to {extracted}")


if __name__ == "__main__":
    main()

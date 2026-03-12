from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from cub_utils import locate_cub_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or extract the CUB-200-2011 dataset from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default="wenewone/cub2002011",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--dest",
        default="data/raw/cub2002011",
        help="Directory that will contain the extracted dataset.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        help="Use a local Kaggle zip file instead of calling the Kaggle API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the destination directory already exists.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, destination: Path, force: bool) -> Path:
    ensure_clean_dir(destination, force=force)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return locate_cub_root(destination)


def find_latest_zip(download_dir: Path) -> Path:
    zip_files = sorted(download_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {download_dir} after Kaggle download.")
    return zip_files[0]


def download_via_kaggle(dataset_slug: str, working_dir: Path) -> Path:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Missing Kaggle API token. Create %USERPROFILE%\\.kaggle\\kaggle.json first, "
            "or download the zip manually from Kaggle and rerun this script with --zip."
        )

    working_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle.cli",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(working_dir),
        "--force",
    ]
    subprocess.run(command, check=True)
    return find_latest_zip(working_dir)


def main() -> int:
    args = parse_args()

    destination = Path(args.dest).resolve()
    download_dir = destination.parent / "_downloads"

    if args.zip_path:
        zip_path = Path(args.zip_path).resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
    else:
        zip_path = download_via_kaggle(args.dataset, download_dir)

    cub_root = extract_zip(zip_path, destination, force=args.force)
    print(f"Dataset extracted to: {destination}")
    print(f"CUB metadata root:    {cub_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

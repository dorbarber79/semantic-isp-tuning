"""Download and fit a resumable, research-only MIT-Adobe FiveK sample."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image
import tifffile

from fivek_photographer_fit import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_FIVEK, IMAGE_IDS, PHOTOGRAPHERS


SOURCE_URL = "https://data.csail.mit.edu/graphics/fivek/img"
RUNNER = Path(__file__).with_name("fivek_photographer_fit.py")
DEFAULT_OUTPUT = DEFAULT_DATA / "results/isp_match_105_v3"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def choose_images(data_root, fivek_root, count, seed):
    licensed = set((data_root / "filesAdobeMIT.txt").read_text().splitlines())
    original = list(IMAGE_IDS)
    if not set(original) <= licensed:
        raise ValueError("An original image is absent from the MIT-licensed list")
    available = sorted(name for name in licensed - set(original)
                       if "/" not in name and "\\" not in name
                       and (fivek_root / "default_isp_linear_720p" / (name + ".png")).is_file())
    if len(available) < count:
        raise ValueError("Not enough licensed images with existing default-ISP renders")
    return original + random.Random(seed).sample(available, count)


def valid_download(path, is_raw):
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with tifffile.TiffFile(path) as source:
            page = source.pages[0]
            if is_raw:
                return bool(page.tags.get("DNGVersion"))
            return page.dtype.name == "uint16" and len(page.shape) == 3 and page.shape[-1] == 3 and bool(page.tags.get("InterColorProfile"))
    except (OSError, ValueError, tifffile.TiffFileError):
        return False


def download(url, destination, is_raw):
    if valid_download(destination, is_raw):
        return
    if destination.exists():
        raise ValueError(f"Existing file is invalid, refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(3):
        temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FiveK-research-sample/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if expected and temporary.stat().st_size != int(expected):
                raise ValueError(f"Incomplete transfer: {url}")
            if not valid_download(temporary, is_raw):
                raise ValueError(f"Downloaded file failed validation: {url}")
            temporary.replace(destination)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            print(f"Download attempt {attempt + 1} failed for {url}: {error}", flush=True)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_image(data_root, image_id):
    encoded = urllib.parse.quote(image_id, safe="")
    download(f"{SOURCE_URL}/dng/{encoded}.dng", data_root / "raw" / f"{image_id}.dng", True)
    for photographer in PHOTOGRAPHERS:
        download(f"{SOURCE_URL}/tiff16_{photographer.lower()}/{encoded}.tif",
                 data_root / "photographers" / photographer / f"{image_id}.tif", False)


def validate_result(folder, image_id):
    result_path = folder / "results.json"
    if not result_path.is_file():
        return None
    data = json.loads(result_path.read_text())
    if data["images"] != [image_id] or set(data["photographers"]) != set(PHOTOGRAPHERS):
        raise ValueError(f"Result identity mismatch: {folder}")
    for row in data["per_image"][image_id].values():
        paths = [Path(row[key]) for key in ("neutral_display_path", "display_path", "fit_display_path")]
        dimensions = []
        for path in paths:
            with Image.open(path) as image:
                dimensions.append(image.size)
        if len(set(dimensions)) != 1:
            raise ValueError(f"Comparison image dimensions differ for {image_id}: {dimensions}")
        for group in ("D1", "D2", "D3", "D4"):
            if len(set(row["descriptors"][group]["variants"])) != 4:
                raise ValueError(f"Expected four distinct {group} variants for {image_id}")
    if not all(Path(path).is_file() for path in data["figures"]):
        raise ValueError(f"Missing comparison figure: {folder}")
    return data


def write_index(output, image_ids):
    completed = []
    csv_rows = {"controls.csv": [], "expert_descriptors.csv": []}
    headers = {}
    for image_id in image_ids:
        folder = output / "images" / image_id
        result = validate_result(folder, image_id) if folder.exists() else None
        if result is None:
            continue
        completed.append({"image_id": image_id, "results": str(folder / "results.json"),
                          "figure": str(folder / "figures" / f"{image_id}_isp_fit_comparison.png")})
        for filename in csv_rows:
            with (folder / filename).open(newline="") as stream:
                reader = csv.DictReader(stream)
                if filename in headers and headers[filename] != reader.fieldnames:
                    raise ValueError(f"CSV schema mismatch: {folder / filename}")
                headers[filename] = reader.fieldnames
                csv_rows[filename].extend(reader)
    for filename, rows in csv_rows.items():
        if not rows:
            continue
        path = output / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers[filename])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    save_json(output / "index.json", {"total_images": len(image_ids), "completed_images": len(completed),
        "completed_pairs": len(completed) * len(PHOTOGRAPHERS), "images": completed,
        "descriptor_note": "Rule-based control-derived drafts, not independent visual annotations or photographer instructions."})
    return len(completed)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(exist_ok=True)
    image_ids = choose_images(args.data_root, args.fivek_root, args.count, args.seed)
    manifest = {"images": image_ids, "existing_images": len(IMAGE_IDS), "additional_images": args.count,
        "seed": args.seed, "photographers": list(PHOTOGRAPHERS), "iterations": args.iterations,
        "learning_rate": args.learning_rate, "device": args.device, "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config), "runner_sha256": sha256(RUNNER),
        "batch_sha256": sha256(Path(__file__)), "license_file": str(args.data_root / "LicenseAdobeMIT.txt"),
        "license_list_sha256": sha256(args.data_root / "filesAdobeMIT.txt"),
        "descriptor_note": "Four template variants per D1-D4, generated from fitted ISP controls; not visual judgments."}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run manifest differs; do not mix changed code or data in an existing output")
    else:
        save_json(manifest_path, manifest)
    status_path = args.out / "progress.json"
    failures = []
    try:
        for index, image_id in enumerate(image_ids, 1):
            folder = images_dir / image_id
            try:
                if folder.exists() and validate_result(folder, image_id) is not None:
                    print(f"[{index}/{len(image_ids)}] already complete: {image_id}", flush=True)
                    continue
                if folder.exists():
                    previous = images_dir / (image_id + ".incomplete-" + uuid.uuid4().hex)
                    folder.rename(previous)
                    print(f"Preserved incomplete run: {previous}", flush=True)
                save_json(status_path, {"state": "downloading", "image_id": image_id, "index": index,
                                        "total": len(image_ids), "failures": failures})
                print(f"[{index}/{len(image_ids)}] preparing {image_id}", flush=True)
                download_image(args.data_root, image_id)
                save_json(status_path, {"state": "fitting", "image_id": image_id, "index": index,
                                        "total": len(image_ids), "failures": failures})
                command = [sys.executable, str(RUNNER), "--data-root", str(args.data_root),
                    "--fivek-root", str(args.fivek_root), "--config", str(args.config), "--out", str(folder),
                    "--image-ids", image_id, "--photographers", *PHOTOGRAPHERS,
                    "--iterations", str(args.iterations), "--learning-rate", str(args.learning_rate),
                    "--device", args.device]
                subprocess.run(command, check=True)
                validate_result(folder, image_id)
                completed = write_index(args.out, image_ids)
                save_json(status_path, {"state": "running", "last_completed": image_id,
                                        "total": len(image_ids), "completed_images": completed, "failures": failures})
                print(f"Completed {completed}/{len(image_ids)} images", flush=True)
            except Exception as error:
                print(f"FAILED {image_id}: {error}", flush=True)
                failures.append({"image_id": image_id, "error": str(error)})
                save_json(status_path, {"state": "running", "total": len(image_ids), "failures": failures})
        completed = write_index(args.out, image_ids)
        state = "completed" if not failures else "completed_with_errors"
        save_json(status_path, {"state": state, "total": len(image_ids), "completed_images": completed, "failures": failures})
        if failures:
            raise RuntimeError(f"{len(failures)} images failed; restart this batch to retry them")
    except Exception as error:
        save_json(status_path, {"state": "failed", "error": str(error), "total": len(image_ids),
                                "failures": failures, "completed_images": write_index(args.out, image_ids)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fivek-root", type=Path, default=DEFAULT_FIVEK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.count < 0 or args.iterations < 1 or args.learning_rate <= 0:
        parser.error("Count must be nonnegative and optimization settings must be positive")
    run(args)


if __name__ == "__main__":
    main()
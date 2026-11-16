"""Acquire the MIT-listed full-resolution FiveK DNG and A-E TIFF masters."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import rawpy
import tifffile


SOURCE_URL = "https://data.csail.mit.edu/graphics/fivek/img"
DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")
LICENSE_SOURCE = Path("/data/data1/FiveK/C_data/LicenseAdobeMIT.txt")
ALLOWLIST_SOURCE = Path("/data/data1/FiveK/C_data/filesAdobeMIT.txt")
PHOTOGRAPHERS = ("A", "B", "C", "D", "E")
USER_AGENT = "MIT-Adobe-FiveK-research-acquisition/1.0"
MINIMUM_BYTES = 1024


class AcquisitionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def copy_input(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise AcquisitionError(f"Required authorization input is not a regular file: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or sha256(source) != sha256(destination):
            raise AcquisitionError(f"Existing authorization copy differs; refusing to overwrite: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".part-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
    except FileExistsError:
        raise AcquisitionError(f"Authorization copy appeared concurrently: {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def load_allowlist(path: Path) -> list[str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError(f"Allowlist is not UTF-8: {error}") from error
    ids = text.splitlines()
    if len(ids) != 2310 or len(set(ids)) != 2310:
        raise AcquisitionError(f"Expected 2310 unique allowlist lines; got {len(ids)} lines")
    if any(not image_id or image_id in (".", "..") or "/" in image_id or "\\" in image_id
           or "\0" in image_id for image_id in ids):
        raise AcquisitionError("Allowlist contains an empty or unsafe source ID")
    return ids


def safe_source_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise AcquisitionError(f"Symlink is forbidden in source tree: {relative}")
    resolved_parent = candidate.parent.resolve()
    if not resolved_parent.is_relative_to(root.resolve()):
        raise AcquisitionError(f"Source path escapes the dataset root: {relative}")
    return candidate


def source_specs(root: Path, image_id: str) -> list[dict[str, str]]:
    encoded = urllib.parse.quote(image_id, safe="")
    specs = [{
        "role": "dng",
        "photographer": "original",
        "relative_path": f"source/dng/{image_id}.dng",
        "url": f"{SOURCE_URL}/dng/{encoded}.dng",
    }]
    specs.extend({
        "role": "tiff16",
        "photographer": photographer,
        "relative_path": f"source/tiff16_{photographer.lower()}/{image_id}.tif",
        "url": f"{SOURCE_URL}/tiff16_{photographer.lower()}/{encoded}.tif",
    } for photographer in PHOTOGRAPHERS)
    for spec in specs:
        spec["path"] = str(safe_source_path(root, spec["relative_path"]))
    return specs


def exiftool_metadata(root: Path, paths: list[Path]) -> dict[str, dict[str, object]]:
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    result = subprocess.run(
        ["exiftool", "-json", "-G1", "-s", "-a", "-n", *relative_paths],
        check=False, capture_output=True, text=True, timeout=180, cwd=root,
    )
    if result.returncode != 0:
        raise AcquisitionError(f"exiftool failed: {result.stderr.strip()[:1000]}")
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AcquisitionError(f"exiftool returned invalid JSON: {error}") from error
    by_path = {}
    for row in rows:
        if "SourceFile" not in row:
            continue
        source_file = Path(row["SourceFile"])
        if source_file.is_absolute():
            source_file = Path(os.path.relpath(source_file, root))
        relative_source = source_file.as_posix()
        row["SourceFile"] = relative_source
        by_path[relative_source] = row
    expected = set(relative_paths)
    if set(by_path) != expected:
        raise AcquisitionError("exiftool did not return metadata for every source file")
    return by_path


def icc_info(page: tifffile.TiffPage) -> dict[str, object]:
    tag = page.tags.get("InterColorProfile")
    if tag is None:
        return {"status": "absent", "byte_length": 0, "sha256": None}
    profile = bytes(tag.value)
    if len(profile) < 128 or profile[36:40] != b"acsp":
        raise AcquisitionError("Embedded TIFF ICC profile is truncated or invalid")
    declared_length = int.from_bytes(profile[:4], "big")
    if declared_length > len(profile) or declared_length < 128:
        raise AcquisitionError("Embedded TIFF ICC profile has an invalid declared length")
    return {
        "status": "present",
        "byte_length": len(profile),
        "sha256": hashlib.sha256(profile).hexdigest(),
    }


def validate_dng(path: Path) -> dict[str, object]:
    with tifffile.TiffFile(path) as source:
        if not source.pages or source.pages[0].tags.get("DNGVersion") is None:
            raise AcquisitionError(f"Not a DNG file: {path.name}")
    with rawpy.imread(str(path)) as raw:
        visible = raw.raw_image_visible
        if visible.ndim != 2 or visible.dtype.kind != "u" or visible.dtype.itemsize != 2:
            raise AcquisitionError(f"Unexpected DNG mosaic type or bit depth: {path.name}")
        height, width = map(int, visible.shape)
        raw_info = {
            "visible_width": width,
            "visible_height": height,
            "container_bits": int(visible.dtype.itemsize * 8),
            "raw_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
            "color_description": raw.color_desc.decode("ascii", errors="replace") if raw.color_desc else None,
            "white_level": int(raw.white_level),
            "black_level_per_channel": [int(value) for value in raw.black_level_per_channel],
            "camera_white_balance": [float(value) for value in raw.camera_whitebalance]
            if raw.camera_whitebalance else None,
        }
    return {
        "format": "DNG",
        "width": width,
        "height": height,
        "channels": 1,
        "sample_type": "unsigned integer",
        "container_bits_per_sample": 16,
        "effective_bit_depth": None,
        "icc_profile": {"status": "not_applicable_to_camera_raw", "byte_length": 0, "sha256": None},
        "rawpy": raw_info,
    }


def validate_tiff(path: Path) -> dict[str, object]:
    with tifffile.TiffFile(path) as source:
        if not source.pages:
            raise AcquisitionError(f"TIFF has no image pages: {path.name}")
        page = source.pages[0]
        if page.photometric is None or page.photometric.name != "RGB":
            raise AcquisitionError(f"Expected RGB TIFF: {path.name}")
        shape = page.shape
        if len(shape) != 3 or shape[-1] != 3 or page.dtype.kind != "u" or page.dtype.itemsize != 2:
            raise AcquisitionError(f"Expected full-resolution uint16 RGB TIFF: {path.name}, {shape}, {page.dtype}")
        if tuple(page.bitspersample) != (16, 16, 16) if isinstance(page.bitspersample, tuple) else page.bitspersample != 16:
            raise AcquisitionError(f"Expected 16 bits per RGB sample: {path.name}")
        icc = icc_info(page)
        if icc["status"] != "present":
            raise AcquisitionError(f"Required embedded ICC profile is absent: {path.name}")
        height, width, channels = map(int, shape)
        return {
            "format": "TIFF",
            "width": width,
            "height": height,
            "channels": channels,
            "sample_type": "unsigned integer",
            "container_bits_per_sample": 16,
            "effective_bit_depth": 16,
            "icc_profile": icc,
        }


def validate_file(spec: dict[str, str], expected_record: dict[str, object] | None) -> dict[str, object]:
    path = Path(spec["path"])
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise AcquisitionError(f"Source is not an independent regular file: {spec['relative_path']}")
    file_stat = path.stat()
    if file_stat.st_nlink != 1:
        raise AcquisitionError(f"Source has unexpected hard links: {spec['relative_path']}")
    if file_stat.st_size < MINIMUM_BYTES:
        raise AcquisitionError(f"Source is too small: {spec['relative_path']} ({file_stat.st_size} bytes)")
    digest = sha256(path)
    if expected_record and digest != expected_record.get("sha256"):
        raise AcquisitionError(f"SHA-256 differs from manifest: {spec['relative_path']}")
    details = validate_dng(path) if spec["role"] == "dng" else validate_tiff(path)
    return {
        "role": spec["role"],
        "photographer": spec["photographer"],
        "path": spec["relative_path"],
        "source_url": spec["url"],
        "size_bytes": file_stat.st_size,
        "sha256": digest,
        "image": details,
    }


def download_file(spec: dict[str, str], expected_record: dict[str, object] | None) -> tuple[dict[str, object], dict[str, object]]:
    destination = Path(spec["path"])
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise AcquisitionError(f"Refusing to overwrite non-regular existing source: {spec['relative_path']}")
        recorded = validate_file(spec, expected_record)
        return recorded, {"status": "existing", "url": spec["url"], "content_length": recorded["size_bytes"]}

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response:
                headers = response.headers
                expected_size = headers.get("Content-Length")
                fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".part-", dir=destination.parent)
                temporary = Path(temporary_name)
                digest = hashlib.sha256()
                byte_count = 0
                with os.fdopen(fd, "wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        byte_count += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if byte_count < MINIMUM_BYTES:
                    raise AcquisitionError(f"Response too small ({byte_count} bytes): {spec['url']}")
                if expected_size is not None and byte_count != int(expected_size):
                    raise AcquisitionError(f"Content-Length mismatch for {spec['url']}: {byte_count} != {expected_size}")
                metadata = validate_dng(temporary) if spec["role"] == "dng" else validate_tiff(temporary)
                if destination.exists() or destination.is_symlink():
                    raise AcquisitionError(f"Refusing to overwrite a concurrently-created file: {spec['relative_path']}")
                os.link(temporary, destination)
                temporary.unlink()
                temporary = None
                size = destination.stat().st_size
                return {
                    "role": spec["role"],
                    "photographer": spec["photographer"],
                    "path": spec["relative_path"],
                    "source_url": spec["url"],
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                    "image": metadata,
                }, {
                    "status": "downloaded",
                    "url": spec["url"],
                    "content_length": int(expected_size) if expected_size is not None else None,
                    "content_type": headers.get("Content-Type"),
                    "last_modified": headers.get("Last-Modified"),
                    "etag": headers.get("ETag"),
                }
        except (OSError, ValueError, urllib.error.URLError, AcquisitionError, tifffile.TiffFileError) as error:
            last_error = error
            print(f"retry {attempt}/3 {spec['relative_path']}: {error}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AcquisitionError(f"Failed source after 3 attempts ({spec['relative_path']}): {last_error}")


def load_manifest(path: Path, authorized_ids: set[str]) -> tuple[dict[str, dict[str, object]], bool]:
    if not path.exists():
        return {}, False
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        last_newline = data.rfind(b"\n")
        with path.open("r+b") as stream:
            stream.truncate(last_newline + 1 if last_newline >= 0 else 0)
            stream.flush()
            os.fsync(stream.fileno())
        data = data[:last_newline + 1] if last_newline >= 0 else b""
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            record = json.loads(line)
            image_id = record["image_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise AcquisitionError(f"Invalid manifest record at line {line_number}: {error}") from error
        if image_id not in authorized_ids or image_id in records:
            raise AcquisitionError(f"Unexpected or duplicate manifest scene at line {line_number}: {image_id!r}")
        if len(record.get("sources", [])) != 6:
            raise AcquisitionError(f"Manifest scene lacks six source records: {image_id}")
        records[image_id] = record
    return records, True


def append_manifest_record(path: Path, record: dict[str, object]) -> None:
    encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise AcquisitionError("Short append to source manifest; incomplete final line will be repaired on resume")
        os.fsync(fd)
    finally:
        os.close(fd)


def normalize_manifest_paths(path: Path, records: dict[str, dict[str, object]]) -> None:
    changed = False
    for record in records.values():
        for source in record.get("sources", []):
            metadata = source.get("exiftool")
            if metadata is not None and metadata.get("SourceFile") != source["path"]:
                metadata["SourceFile"] = source["path"]
                changed = True
    if not changed:
        return
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records.values():
                stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def process_scene(root: Path, image_id: str, prior: dict[str, object] | None) -> dict[str, object]:
    specs = source_specs(root, image_id)
    prior_sources = {source["path"]: source for source in prior.get("sources", [])} if prior else {}
    sources: list[dict[str, object]] = []
    transfers: list[dict[str, object]] = []
    for spec in specs:
        recorded, transfer = download_file(spec, prior_sources.get(spec["relative_path"]))
        sources.append(recorded)
        transfers.append(transfer)

    paths = [Path(spec["path"]) for spec in specs]
    exif = exiftool_metadata(root, paths)
    for source in sources:
        source["exiftool"] = exif[source["path"]]

    return {
        "image_id": image_id,
        "dataset": "MIT-Adobe FiveK, filesAdobeMIT allowlist",
        "license": "MIT-Adobe FiveK research license; copied to ../LicenseAdobeMIT.txt",
        "license_file": "LicenseAdobeMIT.txt",
        "allowlist_file": "filesAdobeMIT.txt",
        "scene_page_metadata": {"status": "unavailable", "page_url": None, "metadata": None},
        "sources": sources,
        "transfer_metadata": transfers,
    }


def write_status(path: Path, *, state: str, total: int, completed: int, errors: list[dict[str, str]],
                 started_at: str, bytes_verified: int, active: list[str] | None = None) -> None:
    atomic_json(path, {
        "state": state,
        "started_at": started_at,
        "updated_at_unix": time.time(),
        "total_scenes": total,
        "completed_scenes": completed,
        "remaining_scenes": total - completed,
        "verified_source_bytes": bytes_verified,
        "active_scenes": active or [],
        "errors": errors,
        "max_concurrency": 4,
    })


def process_tracked_scene(root: Path, image_id: str, prior: dict[str, object] | None,
                          active: set[str], active_lock: threading.Lock) -> dict[str, object]:
    with active_lock:
        active.add(image_id)
    try:
        return process_scene(root, image_id, prior)
    finally:
        with active_lock:
            active.discard(image_id)


def run(root: Path, workers: int) -> int:
    if workers < 1 or workers > 4:
        raise AcquisitionError("Concurrency must be between 1 and 4")
    if not LICENSE_SOURCE.is_file() or not ALLOWLIST_SOURCE.is_file():
        raise AcquisitionError("The external MIT license or allowlist is missing")

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".download_sources.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AcquisitionError(f"Another downloader holds {lock_path}") from error

        copy_input(LICENSE_SOURCE, root / "LicenseAdobeMIT.txt")
        copy_input(ALLOWLIST_SOURCE, root / "filesAdobeMIT.txt")
        ids = load_allowlist(root / "filesAdobeMIT.txt")
        if (root / "filesAdobeMIT.txt").read_bytes() != ALLOWLIST_SOURCE.read_bytes():
            raise AcquisitionError("Copied allowlist is not byte-for-byte identical to its source")

        metadata_dir = root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = metadata_dir / "source_manifest.jsonl"
        status_path = metadata_dir / "acquisition_status.json"
        records, manifest_existed = load_manifest(manifest_path, set(ids))
        normalize_manifest_paths(manifest_path, records)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        errors: list[dict[str, str]] = []
        bytes_verified = sum(int(source["size_bytes"]) for record in records.values()
                             for source in record.get("sources", []))
        write_status(status_path, state="running", total=len(ids), completed=len(records), errors=errors,
                     started_at=started_at, bytes_verified=bytes_verified)
        if not manifest_existed:
            fd = os.open(manifest_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)
            os.fsync(fd)
            os.close(fd)

        pending = [image_id for image_id in ids if image_id not in records]
        print(f"Dataset root: {root}\nScenes: {len(ids)}; complete: {len(records)}; pending: {len(pending)}; workers: {workers}", flush=True)
        active: set[str] = set()
        active_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_tracked_scene, root, image_id, None, active, active_lock): image_id
                       for image_id in pending}
            for future in concurrent.futures.as_completed(futures):
                image_id = futures[future]
                try:
                    record = future.result()
                    append_manifest_record(manifest_path, record)
                    records[image_id] = record
                    bytes_verified += sum(int(source["size_bytes"]) for source in record["sources"])
                    print(f"completed {len(records)}/{len(ids)}: {image_id} ({bytes_verified} verified bytes)", flush=True)
                except Exception as error:
                    errors.append({"image_id": image_id, "error": f"{type(error).__name__}: {error}"})
                    print(f"FAILED {image_id}: {type(error).__name__}: {error}", flush=True)
                write_status(status_path, state="running", total=len(ids), completed=len(records),
                             errors=errors, started_at=started_at, bytes_verified=bytes_verified,
                             active=sorted(active))
        state = "complete" if len(records) == len(ids) and not errors else "incomplete"
        write_status(status_path, state=state, total=len(ids), completed=len(records), errors=errors,
                     started_at=started_at, bytes_verified=bytes_verified)
        print(f"{state}: {len(records)}/{len(ids)} scenes, {len(errors)} scene errors, {bytes_verified} verified bytes", flush=True)
        return 0 if state == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent scenes (1-4)")
    args = parser.parse_args()
    try:
        return run(args.root.resolve(), args.workers)
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

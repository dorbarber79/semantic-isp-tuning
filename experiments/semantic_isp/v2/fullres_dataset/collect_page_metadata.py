"""Extract licensed per-image semantic and camera summaries from the FiveK page."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import urllib.request

from bs4 import BeautifulSoup


DEFAULT_ROOT = Path('/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1')
PAGE_URL = 'https://data.csail.mit.edu/graphics/fivek/'


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
    temporary.replace(path)


def parse_exif_summary(text):
    lens = re.search(r'\b\d+(?:-\d+)?(?:\.\d+)?\s*mm\b', text, flags=re.IGNORECASE)
    aperture = re.search(r'\bF\s*/\s*(\d+(?:\.\d+)?)\b', text, flags=re.IGNORECASE)
    shutter = re.search(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?/\d+(?:\.\d+)?\b', text)
    iso = re.search(r'\bISO\s*(\d+)\b', text, flags=re.IGNORECASE)
    megapixels = re.search(r'\b(\d+(?:\.\d+)?)\s*MP\b', text, flags=re.IGNORECASE)
    camera_end = lens.start() if lens else (aperture.start() if aperture else None)
    return {
        'camera_model_summary': text[:camera_end].strip() if camera_end is not None else None,
        'lens_summary': lens.group(0).strip() if lens else None,
        'aperture_f_number': float(aperture.group(1)) if aperture else None,
        'exposure_time': shutter.group(0) if shutter else None,
        'iso': int(iso.group(1)) if iso else None,
        'megapixels': float(megapixels.group(1)) if megapixels else None,
    }


def extract_page_records(html_bytes, allowed_ids):
    soup = BeautifulSoup(html_bytes, 'html.parser')
    rows = {}
    for row in soup.select('table.data tr'):
        dng_href = next((link.get('href', '') for link in row.select('a[href]')
                         if link.get('href', '').lower().endswith('.dng')), None)
        if not dng_href:
            continue
        image_id = Path(dng_href).stem
        cells = row.find_all('td', recursive=False)
        if not cells:
            raise ValueError(f'FiveK page row has a DNG link but no cells: {image_id}')
        labels = {}
        for cell in cells:
            text = cell.get_text(' ', strip=True)
            label, separator, value = text.partition(':')
            if separator and label.strip() in {'Subject', 'Light', 'Location', 'Time', 'EXIF'}:
                labels[label.strip()] = value.strip()
        if image_id in rows:
            raise ValueError(f'Duplicate image ID on FiveK page: {image_id}')
        rows[image_id] = {
            'image_id': image_id,
            'page_row_number': cells[0].get_text(' ', strip=True),
            'dng_page_href': dng_href,
            'subject': labels.get('Subject'),
            'lighting': labels.get('Light'),
            'location': labels.get('Location'),
            'time_of_day': labels.get('Time'),
            'exif_summary_raw': labels.get('EXIF'),
            'exif_summary_parsed': parse_exif_summary(labels['EXIF']) if labels.get('EXIF') else None,
            'source_fields_present': sorted(labels),
        }
    missing = sorted(set(allowed_ids) - set(rows))
    if missing:
        raise ValueError(f'FiveK webpage has no row for {len(missing)} licensed IDs, e.g. {missing[:5]}')
    return [rows[image_id] for image_id in sorted(allowed_ids)], len(rows)


def run(root, page_path=None):
    root = Path(root).absolute()
    allowlist_path = root / 'filesAdobeMIT.txt'
    if not allowlist_path.is_file() or allowlist_path.is_symlink():
        raise FileNotFoundError(f'Missing copied MIT allowlist: {allowlist_path}')
    allowlist_bytes = allowlist_path.read_bytes()
    allowed_ids = allowlist_bytes.decode('utf-8').splitlines()
    if len(allowed_ids) != 2310 or len(set(allowed_ids)) != 2310:
        raise ValueError('Expected exactly 2,310 unique authorized image IDs')

    if page_path is None:
        request = urllib.request.Request(PAGE_URL, headers={'User-Agent': 'FiveK-research-metadata/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            html_bytes = response.read()
            final_url = response.geturl()
    else:
        page_path = Path(page_path).resolve(strict=True)
        html_bytes = page_path.read_bytes()
        final_url = PAGE_URL

    records, webpage_row_count = extract_page_records(html_bytes, allowed_ids)
    metadata_dir = root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = metadata_dir / 'fivek_page_metadata.jsonl'
    lines = ''.join(json.dumps({
        **record,
        'source': {'name': 'official MIT-Adobe FiveK webpage', 'page_url': final_url,
                  'page_sha256': sha256_bytes(html_bytes), 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()},
    }, ensure_ascii=True, sort_keys=True, allow_nan=False) + '\n' for record in records).encode('utf-8')
    source_record = {
        'schema_version': 1,
        'source_name': 'official MIT-Adobe FiveK webpage',
        'source_page_url': final_url,
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'page_sha256': sha256_bytes(html_bytes),
        'page_snapshot_bytes': len(html_bytes),
        'page_rows_with_dng': webpage_row_count,
        'licensed_ids': len(allowed_ids),
        'licensed_ids_matched': len(records),
        'unlicensed_rows_copied': 0,
        'allowlist_sha256': sha256_bytes(allowlist_bytes),
        'records_file': 'fivek_page_metadata.jsonl',
        'recorded_fields': ['subject', 'lighting', 'location', 'time_of_day',
                            'exif_summary_raw', 'exif_summary_parsed'],
        'note': 'Per-scene page fields are transcribed verbatim from the HTML table. The full 5,000-row HTML and non-allowlisted scene metadata are not copied.',
    }
    atomic_write(jsonl_path, lines)
    atomic_write(metadata_dir / 'fivek_page_metadata_source.json',
                 (json.dumps(source_record, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(source_record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--page-html', type=Path,
                        help='Use a previously fetched official page snapshot instead of fetching the live page')
    args = parser.parse_args()
    run(args.root, args.page_html)


if __name__ == '__main__':
    main()
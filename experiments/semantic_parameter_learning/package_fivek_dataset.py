"""Build a self-contained, hash-indexed FiveK-ISPVA release and dataset statistics."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fivek_clip_svr import write_json
from moran_semantic_isp import CONTROL_NAMES, CONTROL_NEUTRAL


WORKSPACE = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
FIVEK = Path('/data/data1/FiveK')
RESULTS = FIVEK / 'C_data/results'
DEFAULT_DATASET = FIVEK / 'datasets/FiveK-ISPVA'
PRIMARY_RUNS = ('isp_match_105_v4', 'isp_match_20_pilot_v1', 'isp_match_20_pilot_v2')
LATEST_REPORTS = (
    ('fit_comparisons_v22', 'Default ISP versus fitted A-E images', RESULTS / 'isp_match_105_screen_v22'),
    ('d1_text_linear_svm', 'D1 text-only linear SVM predictions with per-DOF errors', RESULTS / 'isp_match_105_screen_d1_text_linear_v1'),
    ('d1_vs_all_descriptors', 'D1 versus D1-D4 linear-regressor comparison', RESULTS / 'isp_match_105_d1_vs_all_d_linear_v1'),
    ('d1_linear_all_controls', 'D1-only linear-model trend results for six controls', RESULTS / 'isp_match_105_d1_text_linear_all_controls_v1'),
    ('d1_brightness_only', 'D1 brightness-only regression', RESULTS / 'isp_match_105_d1_brightness_svr_v1'),
    ('clip_image_text_svr', 'Frozen CLIP image-text SVR baseline', RESULTS / 'isp_match_105_clip_svr_v1'),
    ('mlp_d1', 'Frozen-CLIP MLP versus SVM and median', RESULTS / 'isp_match_125_mlp_d1_v2'),
    ('pilot20_visual_review', '20-scene visual annotation pilot', RESULTS / 'isp_match_20_pilot_v1/screen_report'),
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseStore:
    def __init__(self, root, reuse_objects=None):
        self.root = root
        self.objects = root / 'objects'
        self.objects.mkdir(parents=True, exist_ok=True)
        self.assets = {}
        self.source_cache = {}
        self.reuse_objects = reuse_objects

    def add(self, source, relative, expected_hash=None):
        source = Path(source).resolve()
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Asset paths must remain within the release')
        if not source.is_file():
            raise FileNotFoundError(source)
        key = str(source)
        signature = (source.stat().st_size, source.stat().st_mtime_ns)
        cached = self.source_cache.get(key)
        if cached is not None and cached['signature'] == signature:
            digest = cached['sha256']
        else:
            digest = file_hash(source)
            self.source_cache[key] = {'signature': signature, 'sha256': digest}
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f'Source checksum changed: {source}')
        object_path = self.objects / digest[:2] / digest
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not object_path.exists():
            reusable = self.reuse_objects / digest[:2] / digest if self.reuse_objects else None
            if reusable is not None and reusable.is_file() and file_hash(reusable) == digest:
                os.link(reusable, object_path)
            else:
                temporary = object_path.with_name(digest + '.tmp-' + uuid.uuid4().hex)
                shutil.copyfile(source, temporary)
                if file_hash(temporary) != digest:
                    temporary.unlink()
                    raise ValueError(f'Source changed during snapshot: {source}')
                temporary.replace(object_path)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != digest:
                raise ValueError(f'Conflicting release asset: {relative}')
            if destination.stat().st_ino != object_path.stat().st_ino:
                destination.unlink()
                os.link(object_path, destination)
        else:
            os.link(object_path, destination)
        record = self.assets.setdefault(relative.as_posix(), {
            'path': relative.as_posix(), 'sha256': digest, 'bytes': object_path.stat().st_size,
            'sources': []})
        if key not in record['sources']:
            record['sources'].append(key)
        return relative.as_posix()


def annotation_layers(image_id, expert, documents):
    layers = []
    for source, document in documents:
        for key, target in (('images', 'fitted_output_vs_default'),
                            ('annotations', 'expert_preview_vs_default'),
                            ('names', 'assistant_style_name'),
                            ('comparisons', 'historical_look_reference_and_comparison')):
            value = document.get(key, {}).get(image_id, {}).get(expert)
            if value is None:
                continue
            process = document.get('annotation_process', {})
            layers.append({'source_document': source, 'source_section': key,
                           'annotation_target': target, 'value': value,
                           'method': document.get('method'), 'limitations': document.get('limitations'),
                           'global_annotation_process': process,
                           'scene_provenance': process.get('scene_provenance', {}).get(image_id),
                           'reference_note': document.get('reference_note')})
    return layers


def merge_fit_descriptions(documents):
    images, provenance, metadata = {}, {}, {}
    for source, document in documents:
        descriptions = document.get('images')
        if not isinstance(descriptions, dict) or not descriptions:
            continue
        metadata[source] = {key: value for key, value in document.items() if key != 'images'}
        for image_id, fits in descriptions.items():
            if image_id in images:
                raise ValueError(f'Duplicate fit-annotation scene across sources: {image_id}')
            if set(fits) != set('ABCDE') or any(
                    set(groups) != {'D1', 'D2', 'D3', 'D4'} or
                    any(not isinstance(text, str) or not text.strip() for text in groups.values())
                    for groups in fits.values()):
                raise ValueError(f'Incomplete fit descriptions: {image_id}')
            images[image_id] = fits
            process = document.get('annotation_process', {})
            provenance[image_id] = {'source_document': source,
                'original_scene_provenance': process.get('scene_provenance', {}).get(image_id),
                'source_global_annotation_process': {key: value for key, value in process.items()
                                                     if key != 'scene_provenance'}}
    return {'schema_version': 1,
            'method': 'Lossless consolidation of assistant-authored fit-versus-Default ISP visual descriptions.',
            'limitations': 'Original model/provenance uncertainty is preserved. Reduced-resolution observations are not validated full-resolution sharpness labels.',
            'annotation_process': {'source_metadata': metadata, 'scene_provenance': provenance,
                'snapshot_counts': {'reviewed_images': len(images),
                                    'reviewed_fits': sum(len(fits) for fits in images.values())}},
            'images': images}


def control_statistics(rows):
    summary = {}
    for expert in ('all', *'ABCDE'):
        subset = rows if expert == 'all' else [row for row in rows if row['expert'] == expert]
        if not subset:
            continue
        summary[expert] = {}
        for domain in ('normalized_controls', 'physical_controls'):
            summary[expert][domain] = {}
            for name in CONTROL_NAMES:
                values = np.asarray([row[domain][name] for row in subset], dtype=float)
                summary[expert][domain][name] = {
                    'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
                    'std_population': float(values.std()), 'minimum': float(values.min()),
                    'maximum': float(values.max()), 'q10': float(np.quantile(values, .1)),
                    'q90': float(np.quantile(values, .9))}
    return summary


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_statistics(root, rows):
    output = root / 'statistics'
    output.mkdir()
    summary = control_statistics(rows)
    write_json(output / 'control_summary.json', summary)
    write_csv(output / 'controls_per_pair.csv', [
        {'image_id': row['image_id'], 'expert': row['expert'],
         **{f'{name}_p90': row['normalized_controls'][name] for name in CONTROL_NAMES},
         **{f'{name}_physical': row['physical_controls'][name] for name in CONTROL_NAMES},
         'fit_loss': row['fit_metadata']['loss'],
         'default_vs_expert_mae': row['fit_metadata']['default_isp_target_mae_display'],
         'fit_vs_expert_mae': row['fit_metadata']['fit_target_mae_display']}
        for row in rows])
    flat_summary = []
    for expert, domains in summary.items():
        for domain, controls in domains.items():
            for name, values in controls.items():
                flat_summary.append({'expert': expert, 'domain': domain, 'control': name, **values})
    write_csv(output / 'means_by_expert.csv', flat_summary)
    bins = np.linspace(-1, 1, 41)
    histograms = []
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, CONTROL_NAMES):
        for expert in 'ABCDE':
            values = [row['normalized_controls'][name] for row in rows if row['expert'] == expert]
            counts, edges = np.histogram(values, bins=bins)
            for index, count in enumerate(counts):
                histograms.append({'control': name, 'expert': expert,
                                   'bin_left': float(edges[index]), 'bin_right': float(edges[index + 1]),
                                   'count': int(count)})
            axis.hist(values, bins=bins, histtype='step', linewidth=1.2, label=expert)
        axis.set(title=name, xlabel='Fitted P90-normalized control', ylabel='Number of fits')
        axis.legend(title='Expert')
    figure.savefig(output / 'control_histograms.png', dpi=180)
    plt.close(figure)
    write_csv(output / 'histogram_bins.csv', histograms)
    paired = {image_id: {row['expert']: row for row in rows if row['image_id'] == image_id}
              for image_id in {row['image_id'] for row in rows}}
    differences, distances = [], np.zeros((5, 5))
    for first_index, first in enumerate('ABCDE'):
        for second_index, second in enumerate('ABCDE'):
            valid = [experts for experts in paired.values() if first in experts and second in experts]
            absolute = []
            for name in CONTROL_NAMES:
                delta = np.asarray([experts[first]['normalized_controls'][name] -
                                    experts[second]['normalized_controls'][name] for experts in valid])
                absolute.append(float(np.abs(delta).mean()))
                differences.append({'expert_first': first, 'expert_second': second, 'control': name,
                                    'paired_scenes': len(valid), 'mean_signed_first_minus_second': float(delta.mean()),
                                    'mean_absolute_difference': float(np.abs(delta).mean())})
            distances[first_index, second_index] = np.mean(absolute)
    write_csv(output / 'paired_expert_differences.csv', differences)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(distances, cmap='viridis', vmin=0)
    axis.set(xticks=range(5), yticks=range(5), xticklabels=list('ABCDE'), yticklabels=list('ABCDE'),
             title='Paired expert difference: mean absolute P90 distance')
    for first in range(5):
        for second in range(5):
            axis.text(second, first, f'{distances[first, second]:.3f}', ha='center', va='center', color='white')
    figure.colorbar(image, ax=axis)
    figure.savefig(output / 'expert_difference_heatmap.png', dpi=180)
    plt.close(figure)
    write_json(output / 'expert_distance_matrix.json', {'experts': list('ABCDE'), 'matrix': distances.tolist()})
    coverage = {}
    for row in rows:
        for layer in row['annotation_layers']:
            key = layer['annotation_target']
            coverage.setdefault(key, {'pairs': 0, 'scenes': set()})
            coverage[key]['pairs'] += 1
            coverage[key]['scenes'].add(row['image_id'])
    write_json(output / 'annotation_coverage.json', {
        key: {'layer_records': value['pairs'], 'scenes': len(value['scenes'])}
        for key, value in coverage.items()})
    (output / 'README.md').write_text(
        '# Dataset Statistics\n\nStatistics describe fitted ISP settings, not camera measurements or human ratings.\n'
        'Normalized controls use P90 endpoint units; brightness and contrast have physical neutral 1, others 0.\n'
        'Means are reported separately for normalized and physical controls, and for experts A-E.\n'
        'Expert differences are paired on the same scenes; they do not establish expert quality.\n'
        'Histograms use fixed normalized bins over [-1,1]. Partial annotation coverage is explicit.\n')


def build_unified_gallery(root, rows, merged):
    from fivek_screen_comparisons import draw_comparison
    output = root / 'screen_report'
    output.mkdir()
    pages = []
    for image_id in sorted({row['image_id'] for row in rows}):
        folder = output / 'images' / image_id
        folder.mkdir(parents=True)
        figure_rows = []
        for row in sorted((row for row in rows if row['image_id'] == image_id), key=lambda row: row['expert']):
            notes = merged['images'].get(image_id, {}).get(row['expert'])
            references = []
            style_name = ''
            for layer in row['annotation_layers']:
                if layer['annotation_target'] == 'assistant_style_name':
                    style_name = layer['value']
                if layer['annotation_target'] == 'historical_look_reference_and_comparison':
                    references.extend(layer['value'].get('filter_references', []))
            descriptors = {'fit_based': notes is not None,
                          'groups': {group: {'variants': [notes[group]] if notes else []}
                                     for group in ('D1', 'D2', 'D3', 'D4')},
                          'filter_references': references}
            fit = {**row['fit_metadata'],
                   'neutral_display_path': str(root / row['assets']['default_display_720']['path']),
                   'display_path': str(root / row['assets']['expert_display_720']['path']),
                   'fit_display_path': str(root / row['assets']['fit_display_720']['path'])}
            figure_rows.append({'photographer': row['expert'], 'filter_name': style_name,
                                'fit': fit, 'descriptors': descriptors})
        draw_comparison(folder / 'comparison.png', image_id, figure_rows)
        pages.append(f'<h2>{html.escape(image_id)}</h2>'
                     f'<img style="max-width:100%" src="images/{image_id}/comparison.png">')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Unified FiveK-ISPVA annotated fits</title><h1>Unified FiveK-ISPVA annotated fits</h1>'
        '<p>Default ISP, expert targets and fitted outputs. D1-D4 describe the fitted look.</p>'
        + '\n'.join(pages))


def image_dimensions(path):
    try:
        with Image.open(path) as image:
            return list(image.size)
    except (OSError, ValueError):
        return None


def bundle_latest_reports(release, store):
    report_root = release / 'reports/latest'
    report_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for slug, title, source in LATEST_REPORTS:
        if not source.is_dir():
            raise FileNotFoundError(f'Missing latest report source: {source}')
        destination = report_root / slug
        count = 0
        for path in sorted(source.rglob('*')):
            if path.is_file() and not path.name.startswith('.'):
                store.add(path, Path('reports/latest') / slug / path.relative_to(source))
                count += 1
        if count == 0:
            raise ValueError(f'No report files found in {source}')
        index_path = destination / 'index.html'
        if not index_path.exists():
            useful = ('metrics.json', 'all_controls_metrics.json', 'predictions.csv',
                      'predictions.json', 'out_of_fold_predictions.csv',
                      'out_of_fold_by_control.png', 'all_D_text_embeddings.npz')
            links = [name for name in useful if (destination / name).is_file()]
            index_path.write_text('<!doctype html><meta charset="utf-8">'
                f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><ul>'
                + ''.join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in links)
                + '</ul>')
            store.add(index_path, Path('reports/latest') / slug / 'index.html')
        entries.append({'id': slug, 'title': title, 'source': str(source),
                        'files': count + int(not any(path.name == 'index.html' for path in source.iterdir()))})
        print(f'Bundled latest report {slug}: {count} source files', flush=True)
    (report_root / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>FiveK-ISPVA Current Reports</title>'
        '<h1>Current FiveK-ISPVA Reports</h1><ul>'
        + ''.join(f'<li><a href="{entry["id"]}/index.html">{html.escape(entry["title"])}</a>'
                  f' ({entry["files"]} files)</li>' for entry in entries) + '</ul>')
    store.add(report_root / 'index.html', 'reports/latest/index.html')
    return entries


def build(args):
    destination = args.dataset / 'releases' / args.release
    if Path(args.release).name != args.release or args.release in ('', '.', '..'):
        raise ValueError('Invalid release name')
    if destination.exists():
        raise FileExistsError(f'Immutable release already exists: {destination}')
    release = destination.with_name('.' + args.release + '.building-' + uuid.uuid4().hex)
    release.mkdir(parents=True)
    store = ReleaseStore(release, getattr(args, 'reuse_objects', None))
    documents = []
    for path in sorted(EXPERIMENT.glob('*.json')):
        relative = store.add(path, Path('annotations/source_documents/repository') / path.name)
        document = json.loads((release / relative).read_text())
        documents.append((relative, document))
    for pilot_name in ('isp_match_20_pilot_v1', 'isp_match_20_pilot_v2'):
        pilot_document = RESULTS / pilot_name / 'fit_visual_descriptions.json'
        if pilot_document.exists():
            relative = store.add(pilot_document,
                Path('annotations/source_documents/pilot') / pilot_name / 'fit_visual_descriptions.json')
            documents.append((relative, json.loads((release / relative).read_text())))
    for path in EXPERIMENT.iterdir():
        if path.is_file() and path.suffix in ('.py', '.md'):
            store.add(path, Path('provenance/code/semantic_parameter_learning') / path.name)
    for path in (WORKSPACE / 'experiments/semantic_isp').iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            store.add(path, Path('provenance/code/semantic_isp') / path.name)
    for path in (WORKSPACE / 'papers/semantic_isp_tip.tex',
                 WORKSPACE / '.github/agents/fivek-visual-annotator.agent.md',
                 WORKSPACE / 'docs/SEMANTIC_ISP_RESEARCH.md', WORKSPACE / 'docs/dof scale per word.xlsx'):
        if path.is_file():
            store.add(path, Path('provenance/project') / path.relative_to(WORKSPACE))
    for path in (FIVEK / 'C_data').glob('*MIT*.txt'):
        store.add(path, Path('licenses') / path.name)
    rows, configs, seen = [], set(), set()
    for run in PRIMARY_RUNS:
        run_root = RESULTS / run
        if not (run_root / 'index.json').exists():
            continue
        index = json.loads((run_root / 'index.json').read_text())
        for entry in index['images']:
            image_id = entry['image_id']
            if image_id in seen:
                raise ValueError(f'Scene occurs in two primary runs: {image_id}')
            seen.add(image_id)
            source = json.loads(Path(entry['results']).read_text())
            config = Path(source['calibration_config'])
            configs.add(config)
            config_relative = store.add(config, Path('calibrations') / source['calibration_config_sha256'] / config.name,
                                        source['calibration_config_sha256'])
            for expert, fit in source['per_image'][image_id].items():
                folder = Path('scenes') / image_id
                assets = {}
                fields = (
                    ('raw_original', 'raw_source', folder / 'original/raw.dng', 'raw_source_sha256'),
                    ('expert_original', 'source_tiff', folder / f'experts/{expert}/original.tif', 'source_tiff_sha256'),
                    ('default_linear_720', 'neutral_source', folder / 'default/linear_720.png', 'neutral_source_sha256'),
                    ('default_display_720', 'neutral_display_path', folder / 'default/display_720.png', None),
                    ('expert_display_720', 'display_path', folder / f'experts/{expert}/display_720.png', None),
                    ('fit_display_720', 'fit_display_path', folder / f'experts/{expert}/fit_720.png', None))
                for role, key, relative, checksum in fields:
                    source_path = Path(fit[key])
                    assets[role] = {'path': store.add(source_path, relative, fit.get(checksum) if checksum else None),
                                    'dimensions_wh': image_dimensions(source_path)}
                rows.append({'image_id': image_id, 'expert': expert, 'primary_fit_run': run,
                    'assets': assets, 'calibration': config_relative,
                    'normalized_controls': fit['normalized_controls'], 'physical_controls': fit['physical_controls'],
                    'fit_metadata': fit, 'fit_metadata_original_paths_retained': True,
                    'annotation_layers': annotation_layers(image_id, expert, documents),
                    'resolution_caveat': 'Fit and linguistic detail/sharpness review used 720-pixel-high saved renders and smaller sheets; no original-resolution sharpness validation.'})
            print(f'Packaged primary scene {image_id}', flush=True)
    for config in configs:
        metadata = json.loads(config.read_text())
        roots = {config.parent}
        roots.update(Path(path) for path in metadata.get('centroid_calibration', {}).get('record_roots', []))
        for root in roots:
            if root.is_dir():
                identifier = root.name + '-' + hashlib.sha256(str(root).encode()).hexdigest()[:8]
                for path in sorted(root.rglob('*')):
                    if path.is_file() and path.suffix in ('.json', '.csv', '.pt', '.md'):
                        store.add(path, Path('archives/calibration_runs') / identifier / path.relative_to(root))
    archives = sorted(path for path in RESULTS.glob('isp_match*') if path.is_dir())
    for archive in archives:
        print(f'Archiving {archive.name}', flush=True)
        for path in sorted(archive.rglob('*')):
            if path.is_file() and not path.name.startswith('.'):
                store.add(path, Path('archives/experiments') / archive.name / path.relative_to(archive))
    historical = Path('/data/data1/semantic_isp/results/semantic_parameter_learning')
    if historical.is_dir():
        print('Archiving historical semantic-parameter experiments', flush=True)
        for path in sorted(historical.rglob('*')):
            if path.is_file() and not path.name.startswith('.'):
                store.add(path, Path('archives/semantic_parameter_learning') / path.relative_to(historical))
    latest_reports = bundle_latest_reports(release, store)
    write_json(release / 'annotations/normalized_pairs.json', rows)
    merged = merge_fit_descriptions(documents)
    write_json(release / 'annotations/fit_visual_descriptions.json', merged)
    build_statistics(release, rows)
    build_unified_gallery(release, rows, merged)
    assets = sorted(store.assets.values(), key=lambda value: value['path'])
    write_json(release / 'assets.json', assets)
    source_map = {}
    for asset in assets:
        for source in asset['sources']:
            source_map.setdefault(source, []).append(asset['path'])
    write_json(release / 'source_path_map.json', source_map)
    reviewed = {row['image_id'] for row in rows if any(
        layer['annotation_target'] == 'fitted_output_vs_default' for layer in row['annotation_layers'])}
    manifest = {'dataset_name': 'FiveK ISP Visual Annotation Dataset (FiveK-ISPVA)',
        'release': args.release, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scenes': len(seen), 'expert_fit_pairs': len(rows), 'visually_annotated_scenes': len(reviewed),
        'pending_visual_scenes': len(seen - reviewed), 'source_archives': [path.name for path in archives],
        'storage': 'Source files are copied into content-addressed objects; verified objects from earlier immutable releases may be hardlinked for cross-release deduplication. No source symlinks or links to mutable source assets.',
        'normalized_pairs': 'annotations/normalized_pairs.json', 'assets_index': 'assets.json',
        'merged_fit_descriptions': 'annotations/fit_visual_descriptions.json',
        'unified_visual_report': 'screen_report/index.html',
        'full_resolution': 'Original DNG and expert TIFF masters retained. Full-resolution default RGB and full-resolution fitted outputs were not collected; no upscaled substitute is called full resolution.',
        'annotation_resolution': 'Existing 720-pixel-high renders and smaller comparison-sheet previews; per-scene pilot provenance retained verbatim.',
        'sharpness': 'Existing sharpness/detail phrases and tests are reduced-resolution observations, not verified full-resolution sharpness labels.',
        'ai_provenance': 'Preserved per source and scene. Original model unknown; user-reported GPT-5.6 Luna screenshot is verification-only evidence. Pilot model identifiers remain unavailable where recorded as such.',
        'publication_status': 'Internal release candidate; original licensing terms preserved. Redistribution rights and author/annotation validation must be checked before public release.',
        'latest_reports': latest_reports,
        'objects': len({asset['sha256'] for asset in assets}), 'logical_files': len(assets),
        'unique_asset_bytes': sum((release / 'objects' / digest[:2] / digest).stat().st_size
                                  for digest in {asset['sha256'] for asset in assets})}
    write_json(release / 'dataset_manifest.json', manifest)
    checksums = []
    for asset in assets:
        checksums.append(f"{asset['sha256']}  {asset['path']}")
    for path in [release / 'annotations/normalized_pairs.json', release / 'annotations/fit_visual_descriptions.json', release / 'assets.json',
                 release / 'source_path_map.json', release / 'dataset_manifest.json',
                 *sorted((release / 'statistics').rglob('*')), *sorted((release / 'screen_report').rglob('*')),
                 *sorted((release / 'reports/latest').rglob('*'))]:
        if path.is_file():
            checksums.append(f'{file_hash(path)}  {path.relative_to(release).as_posix()}')
    (release / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    (release / 'README.md').write_text(
        '# FiveK-ISPVA Release\n\nStart with `dataset_manifest.json`, `annotations/normalized_pairs.json`, and `statistics/`.\n'
        'Each pair links original RAW/expert TIFF, linear/default/expert/fit previews, physical and normalized controls, '
        'full fitting metadata, all annotation layers, and calibration provenance.\n'
        'Source annotations remain lossless in `annotations/source_documents`; historical experiments and captions remain in `archives/`.\n'
        'Do not merge control-derived templates or model predictions into visual ground truth.\n'
        'Original absolute paths inside legacy files are preserved; `source_path_map.json` resolves packaged paths.\n'
        'Media and language were collected at reduced resolution for annotation: saved renders are 720 pixels high, '
        'with smaller sheet previews. Full-resolution sharpness and full-resolution fit outputs were not validated/collected.\n'
        'AI identifiers are provenance statements, not guesses. Human validation and redistribution clearance remain outstanding.\n'
        'This is an immutable snapshot. Build a new named release after additional annotation; never edit object files in place.\n')
    (release / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FiveK-ISPVA</title>'
        f'<h1>FiveK-ISPVA: {html.escape(args.release)}</h1><p>{len(seen)} scenes; {len(rows)} expert-fit pairs; '
        f'{len(reviewed)} visually annotated scenes.</p>'
        '<p><a href="dataset_manifest.json">Manifest</a> | <a href="reports/latest/index.html">All current reports</a> | '
        '<a href="annotations/normalized_pairs.json">Pairs and annotations</a> | '
        '<a href="annotations/fit_visual_descriptions.json">Unified D1-D4 descriptions</a> | '
        '<a href="screen_report/index.html">Unified visual report</a> | '
        '<a href="statistics/control_summary.json">Control statistics</a></p>'
        '<img style="max-width:100%" src="statistics/control_histograms.png">'
        '<img style="max-width:100%" src="statistics/expert_difference_heatmap.png">')
    verification = verify_release(release, full_checksums=True)
    print(f"Verified staged release: {verification['assets']} assets, {verification['pairs']} pairs", flush=True)
    release.rename(destination)
    latest = args.dataset / 'latest.json'
    temporary_latest = latest.with_suffix('.json.tmp')
    write_json(temporary_latest, {'release': args.release, 'path': f'releases/{args.release}'})
    temporary_latest.replace(latest)
    psnr_analysis = args.dataset / 'analyses/fit_psnr_trimmed_5pct_v1/index.html'
    psnr_link = ('<li><a href="analyses/fit_psnr_trimmed_5pct_v1/index.html">'
                 'PSNR-trimmed fit and control statistics</a></li>' if psnr_analysis.is_file() else '')
    (args.dataset / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>FiveK-ISPVA Dataset Hub</title>'
        f'<h1>FiveK-ISPVA Dataset Hub</h1><p>Current release: {html.escape(args.release)}</p>'
        f'<ul><li><a href="releases/{args.release}/index.html">Dataset overview and data</a></li>'
        f'<li><a href="releases/{args.release}/reports/latest/index.html">All latest figures, comparisons and model results</a></li>'
        f'<li><a href="releases/{args.release}/statistics/README.md">Dataset statistics and expert comparisons</a></li>'
        f'<li><a href="releases/{args.release}/dataset_manifest.json">Manifest, provenance, resolution and licensing</a></li>'
        f'{psnr_link}</ul>')
    print(json.dumps(manifest, indent=2))


def verify_release(root, full_checksums=False):
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    pairs = json.loads((root / 'annotations/normalized_pairs.json').read_text())
    assets = json.loads((root / 'assets.json').read_text())
    if len(pairs) != manifest['expert_fit_pairs']:
        raise ValueError('Pair count differs from manifest')
    scenes = {row['image_id'] for row in pairs}
    if len(scenes) != manifest['scenes']:
        raise ValueError('Scene count differs from manifest')
    if manifest.get('merged_fit_descriptions'):
        merged = json.loads((root / manifest['merged_fit_descriptions']).read_text())
        if len(merged['images']) != manifest['visually_annotated_scenes']:
            raise ValueError('Merged description coverage differs from manifest')
        for row in pairs:
            if row['image_id'] in merged['images']:
                matched = [layer['value'] for layer in row['annotation_layers']
                           if layer['annotation_target'] == 'fitted_output_vs_default']
                if matched != [merged['images'][row['image_id']][row['expert']]]:
                    raise ValueError('Merged descriptions differ from the preserved source layers')
    for image_id in scenes:
        if {row['expert'] for row in pairs if row['image_id'] == image_id} != set('ABCDE'):
            raise ValueError(f'Incomplete expert coverage: {image_id}')
    digests = {}
    for asset in assets:
        path = root / asset['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != asset['bytes']:
            raise ValueError(f'Missing/changed asset: {path}')
        object_path = root / 'objects' / asset['sha256'][:2] / asset['sha256']
        if not object_path.is_file() or path.stat().st_ino != object_path.stat().st_ino:
            raise ValueError(f'Asset is not linked to its content-addressed object: {path}')
        if full_checksums:
            if asset['sha256'] not in digests:
                digests[asset['sha256']] = file_hash(object_path)
            if digests[asset['sha256']] != asset['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
    summary = control_statistics(pairs)
    saved_summary = json.loads((root / 'statistics/control_summary.json').read_text())
    if summary != saved_summary:
        raise ValueError('Stored control statistics differ from pair records')
    if full_checksums:
        for line in (root / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            if expected not in digests:
                if file_hash(root / relative) != expected:
                    raise ValueError(f'Generated file checksum mismatch: {relative}')
    return {'release': str(root), 'scenes': len(scenes), 'pairs': len(pairs),
            'assets': len(assets), 'full_checksums': full_checksums, 'statistics_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--release', default='2026-11-14-v1')
    parser.add_argument('--reuse-objects', type=Path)
    parser.add_argument('--verify-release', type=Path)
    parser.add_argument('--full-checksums', action='store_true')
    args = parser.parse_args()
    stop = threading.Event()

    def progress():
        while not stop.wait(10):
            print('Dataset packaging is active; copying and validating snapshot assets.', flush=True)

    heartbeat = threading.Thread(target=progress, daemon=True)
    heartbeat.start()
    try:
        if args.verify_release:
            print(json.dumps(verify_release(args.verify_release, args.full_checksums), indent=2))
        else:
            build(args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
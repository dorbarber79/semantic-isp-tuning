"""Compose six independent D1 text-only linear SVRs into a FiveK screen report."""

import argparse
import csv
import html
import json
from pathlib import Path

import joblib
import numpy as np
import open_clip
from PIL import Image
import torch
import torch.nn.functional as F

from fivek_clip_svr import write_json
from fivek_preview import load_linear_png, write_render
from fivek_screen_comparisons import draw_comparison, load_fit_visual_descriptions
from fivek_semantic_descriptions import sha256
from semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints


DEFAULT_SOURCE = Path('/data/data1/FiveK/C_data/results/isp_match_105_v4')
DEFAULT_MODELS = Path('/data/data1/FiveK/C_data/results/isp_match_105_d1_text_linear_all_controls_v1')
DEFAULT_OUT = Path('/data/data1/FiveK/C_data/results/isp_match_105_screen_d1_text_linear_v1')


def prediction_scores(target_controls, predicted_controls, target_pixels, predicted_pixels):
    target = np.asarray(target_controls, dtype=float)
    predicted = np.asarray(predicted_controls, dtype=float)
    if target.shape != (6,) or predicted.shape != (6,):
        raise ValueError('Require six target and predicted controls')
    if target_pixels.shape != predicted_pixels.shape:
        raise ValueError('Image target and prediction dimensions must match')
    difference = np.asarray(predicted_pixels, dtype=float) - np.asarray(target_pixels, dtype=float)
    mse = float(np.mean(difference ** 2))
    return {'parameter_mae': float(np.mean(np.abs(predicted - target))),
            'parameter_rmse': float(np.sqrt(np.mean((predicted - target) ** 2))),
            'parameter_absolute_errors': dict(zip(CONTROL_NAMES, np.abs(predicted - target).tolist())),
            'image_mae_255': float(np.mean(np.abs(difference))),
            'image_psnr_db': float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else None}


def report_rows(result_rows):
    return [{'photographer': row['expert'],
             'title': f"Expert {row['expert']} | D1 text-only linear SVM (SVR)",
             'prediction_display_path': row['render'], 'input_text': row['text'], 'scores': row['scores'],
             'target_controls': row['target_normalized_controls'], 'descriptors': {'fit_based': True},
             'fit': {**row['fit'], 'normalized_controls': row['normalized_controls']}}
            for row in result_rows]


def draw_report(folder, image_id, result_rows):
    draw_comparison(folder / 'comparison.png', image_id, report_rows(result_rows),
        heading=f'{image_id} | Six independent text-only linear SVMs (SVR) | D1 only',
        subheading='Default ISP once | Expert TIFF / target fit / prediction | '
                   'Per-DOF absolute P90 errors | GT and predicted controls')


def refresh_report(output):
    paths = sorted((output / 'images').glob('*/predictions.json'))
    if not paths:
        raise FileNotFoundError(f'No saved predictions in {output}')
    errors_by_pair = {}
    for path in paths:
        result = json.loads(path.read_text())
        for row in result['rows']:
            errors = {name: abs(row['normalized_controls'][name] - row['target_normalized_controls'][name])
                      for name in CONTROL_NAMES}
            row['scores']['parameter_absolute_errors'] = errors
            errors_by_pair[result['image_id'], row['expert']] = errors
        draw_report(path.parent, result['image_id'], result['rows'])
        write_json(path, result)
    csv_path = output / 'predictions.csv'
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    for row in csv_rows:
        errors = errors_by_pair[row['image_id'], row['expert']]
        row.update({f'{name}_absolute_error_p90': value for name, value in errors.items()})
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['estimator_family'] = 'SVM regression (SVR), separate linear model for each DOF'
    manifest['per_control_error'] = 'Absolute predicted-minus-target difference in P90 units'
    manifest['per_control_mae_all_pairs'] = {
        name: float(np.mean([errors[name] for errors in errors_by_pair.values()])) for name in CONTROL_NAMES}
    manifest['report_renderer_sha256'] = sha256(Path(__file__))
    write_json(manifest_path, manifest)
    index_path = output / 'index.html'
    index_text = index_path.read_text()
    index_text = index_text.replace('D1 independent linear models', 'D1 linear SVM regression (SVR)')
    index_text = index_text.replace('D1: six independent linear models', 'D1: six independent linear SVM regressors (SVR)')
    index_path.write_text(index_text)
    print(f'Refreshed {len(paths)} figures and per-DOF errors; no training or prediction changes')


def load_models(root, image_ids):
    models, assignments, hashes = {}, {}, {}
    reference_assignment = None
    for control in CONTROL_NAMES:
        folder = root / control
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['control'] != control or manifest['methods'] != ['text_linear', 'median_baseline']:
            raise ValueError(f'Unexpected experiment for {control}')
        assignment = {}
        for fold in json.loads((folder / 'folds.json').read_text()):
            if set(fold['train_scenes']) & set(fold['test_scenes']):
                raise ValueError(f'Scene leakage in {control} fold {fold["fold"]}')
            path = folder / 'models' / f'text_linear_fold_{fold["fold"]}.joblib'
            model = joblib.load(path)
            if model.kernel != 'linear' or model.n_features_in_ != 512:
                raise ValueError('Require text-only 512-D linear SVRs')
            models[control, fold['fold']] = model
            hashes[str(path)] = sha256(path)
            for image_id in fold['test_scenes']:
                if image_id in assignment:
                    raise ValueError(f'Duplicate held-out scene: {image_id}')
                assignment[image_id] = fold['fold']
        if set(assignment) != set(image_ids):
            raise ValueError(f'Held-out scene coverage differs for {control}')
        if reference_assignment is not None and assignment != reference_assignment:
            raise ValueError('Controls use different outer scene folds')
        reference_assignment = assignment
        assignments[control] = assignment
    return models, assignments, hashes


def run(args):
    index = json.loads((args.source / 'index.json').read_text())
    description_document = load_fit_visual_descriptions()
    image_ids = [entry['image_id'] for entry in index['images']]
    models, assignments, model_hashes = load_models(args.models, image_ids)
    records, texts = [], []
    config_path = None
    for entry in index['images']:
        result = json.loads(Path(entry['results']).read_text())
        if config_path is None:
            config_path = Path(result['calibration_config'])
            config_hash = result['calibration_config_sha256']
        if result['calibration_config_sha256'] != config_hash:
            raise ValueError('Cannot compose controls from mixed fit calibrations')
        for expert in 'ABCDE':
            text = description_document['images'][entry['image_id']][expert]['D1']
            records.append({'image_id': entry['image_id'], 'expert': expert,
                            'fit': result['per_image'][entry['image_id']][expert],
                            'text': text, 'source_result': entry['results'],
                            'source_result_sha256': sha256(Path(entry['results']))})
            texts.append(text)
    if sha256(config_path) != config_hash:
        raise ValueError('Fit calibration has changed')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    endpoints = pole_endpoints(config)
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True, device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            features = model.encode_text(tokenizer(texts[start:start + 64]).to(args.device)).float()
            embeddings.extend(F.normalize(features, dim=-1).cpu().numpy())
    features = np.asarray(embeddings)
    np.savez_compressed(args.out / 'd1_text_embeddings.npz', features=features)
    del model
    predictions = np.zeros((len(records), 6))
    for record_index, record in enumerate(records):
        for column, control in enumerate(CONTROL_NAMES):
            fold = assignments[control][record['image_id']]
            predictions[record_index, column] = np.clip(
                models[control, fold].predict(features[record_index:record_index + 1])[0], -1, 1)
    cached_root = Path(json.loads((args.models / 'warmth/manifest.json').read_text())['source'])
    selected = {row['image_id']: row['expert']
                for row in json.loads((cached_root / 'selected_pairs.json').read_text())}
    csv_rows, pages, aggregate = [], [], []
    for scene_index, entry in enumerate(index['images']):
        image_id = entry['image_id']
        folder = args.out / 'images' / image_id
        folder.mkdir(parents=True)
        scene_records = records[scene_index * 5:(scene_index + 1) * 5]
        neutral = load_linear_png(scene_records[0]['fit']['neutral_source']).to(args.device)
        result_rows = []
        for expert_index, record in enumerate(scene_records):
            record_index = scene_index * 5 + expert_index
            values = predictions[record_index]
            physical = normalized_to_physical(torch.as_tensor(values, device=args.device, dtype=torch.float32), endpoints)
            with torch.inference_mode():
                rendered = render_linear(neutral, parameters_from_controls(physical, config))
            path = folder / f"expert_{record['expert']}_prediction.png"
            write_render(path, rendered)
            with Image.open(record['fit']['fit_display_path']) as image:
                target_pixels = np.asarray(image.convert('RGB'))
            with Image.open(path) as image:
                predicted_pixels = np.asarray(image.convert('RGB'))
            target = [record['fit']['normalized_controls'][name] for name in CONTROL_NAMES]
            scores = prediction_scores(target, values, target_pixels, predicted_pixels)
            predicted_controls = dict(zip(CONTROL_NAMES, values.tolist()))
            result_row = {**record, 'fold': assignments['warmth'][image_id],
                          'normalized_controls': predicted_controls,
                          'physical_controls': dict(zip(CONTROL_NAMES, physical.cpu().tolist())),
                          'target_normalized_controls': record['fit']['normalized_controls'],
                          'render': str(path), 'scores': scores,
                          'previous_selected_pair': selected[image_id] == record['expert']}
            result_rows.append(result_row)
            aggregate.append(result_row)
            csv_rows.append({'image_id': image_id, 'expert': record['expert'], 'fold': result_row['fold'],
                'D1': record['text'], **{key: value for key, value in scores.items() if key != 'parameter_absolute_errors'},
                **{f'{name}_absolute_error_p90': value for name, value in scores['parameter_absolute_errors'].items()},
                **{f'{name}_predicted': value for name, value in predicted_controls.items()},
                **{f'{name}_target': value for name, value in record['fit']['normalized_controls'].items()}})
        draw_report(folder, image_id, result_rows)
        write_json(folder / 'predictions.json', {'image_id': image_id, 'rows': result_rows})
        pages.append(f'<h2>{html.escape(image_id)}</h2>'
                     f'<a href="images/{image_id}/predictions.json">Prediction data</a><br>'
                     f'<img style="max-width:100%" src="images/{image_id}/comparison.png">')
        print(f'Rendered {image_id}: experts A-E', flush=True)
    with (args.out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summaries = {}
    for label, subset in [('all_525_pairs', aggregate),
                          ('previous_selected_105_pairs', [row for row in aggregate if row['previous_selected_pair']])]:
        summaries[label] = {'pairs': len(subset),
            **{key: float(np.mean([row['scores'][key] for row in subset]))
               for key in ('parameter_mae', 'parameter_rmse', 'image_mae_255')},
            'mean_image_psnr_db': float(np.mean([row['scores']['image_psnr_db'] for row in subset
                                                if row['scores']['image_psnr_db'] is not None]))}
    write_json(args.out / 'manifest.json', {
        'method': 'Six independent D1 text-only linear SVRs, composed in the ISP',
        'source_fit_run': str(args.source), 'models': str(args.models), 'model_sha256': model_hashes,
        'calibration_sha256': config_hash, 'source_code_sha256': sha256(Path(__file__)),
        'scenes': len(image_ids), 'experts_per_scene': 5, 'prediction_pairs': len(records),
        'text': 'D1 only, L2-normalized OpenAI CLIP ViT-B/32 embedding; no image embedding',
        'evaluation': 'Each scene predicted by its outer held-out fold; models trained on one selected expert per training scene',
        'parameter_score': 'Mean absolute error across six P90 controls; lower is better',
        'image_score': 'MAE on saved 8-bit sRGB pixels vs target fit (lower better) and PSNR (higher better)',
        'summary': summaries})
    (args.out / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>D1 independent linear models</title><h1>D1: six independent linear models</h1>'
        '<p><a href="manifest.json">Metrics and protocol</a> | '
        '<a href="predictions.csv">Controls and scores</a></p>' + '\n'.join(pages))
    print(json.dumps({'output': str(args.out), 'summary': summaries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--refresh-report-only', action='store_true')
    args = parser.parse_args()
    if args.refresh_report_only:
        refresh_report(args.out)
    else:
        run(args)


if __name__ == '__main__':
    main()
"""Predict FiveK ISP controls from Default ISP images and visual descriptions."""

import argparse
import csv
import html
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn.functional as F

from fivek_preview import load_linear_png, write_render
from fivek_screen_comparisons import (
    FIT_VISUAL_DESCRIPTIONS, draw_comparison, fit_descriptor_record,
    load_fit_references, load_fit_visual_descriptions,
)
from fivek_semantic_descriptions import GROUPS, sha256
from semantic_isp import CONTROL_NAMES, linear_to_srgb, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints


DEFAULT_SOURCE = Path('/data/data1/FiveK/C_data/results/isp_match_105_v4')
DEFAULT_OUTPUT = Path('/data/data1/FiveK/C_data/results/isp_match_105_clip_svr_v1')
VARIANTS = (*GROUPS, 'combined')


def select_experts(image_ids, seed=0):
    ordered = sorted(image_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError('Image IDs must be unique')
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return {ordered[int(position)]: 'ABCDE'[rank % 5]
            for rank, position in enumerate(permutation)}


def descriptor_inputs(groups, style_name='', references=()):
    if set(groups) != set(GROUPS) or any(not groups[group].strip() for group in GROUPS):
        raise ValueError('Require nonempty D1-D4 fit descriptions')
    inputs = {group: groups[group] for group in GROUPS}
    combined = ' '.join(groups[group] for group in GROUPS)
    if style_name:
        combined += f' Look: {style_name}.'
    if references:
        combined += ' Approximate style references: ' + '; '.join(
            f"{item['app']} {item['name']}" for item in references) + '.'
    inputs['combined'] = combined
    return inputs


def fused_features(image_features, text_features):
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError('Require paired image/text feature matrices of equal shape')
    image_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
    if not np.isfinite(image_features).all() or not np.isfinite(text_features).all():
        raise ValueError('CLIP features must be finite')
    if (image_norms <= 0).any() or (text_norms <= 0).any():
        raise ValueError('CLIP feature norms must be positive')
    return np.concatenate((image_features / image_norms, text_features / text_norms), axis=1)


def scene_folds(groups, splits=5, seed=0):
    scenes = np.unique(groups)
    for training, testing in KFold(n_splits=splits, shuffle=True, random_state=seed).split(scenes):
        yield (np.flatnonzero(np.isin(groups, scenes[training])),
               np.flatnonzero(np.isin(groups, scenes[testing])))


def make_regressor():
    return MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale'))


def encode_records(records, output, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True, device=device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    image_features, text_features = [], []
    text_rows = [record['inputs'][variant] for record in records for variant in VARIANTS]
    token_counts = [len(tokenizer.encode(text)) for text in text_rows]
    with torch.inference_mode():
        for record in records:
            with Image.open(record['fit']['neutral_display_path']) as image:
                inputs = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
            features = model.encode_image(inputs).float().cpu().numpy()
            image_features.extend([features[0]] * len(VARIANTS))
        for start in range(0, len(text_rows), 64):
            tokens = tokenizer(text_rows[start:start + 64]).to(device)
            text_features.extend(model.encode_text(tokens).float().cpu().numpy())
    raw_image = np.asarray(image_features)
    raw_text = np.asarray(text_features)
    features = fused_features(raw_image, raw_text)
    np.savez_compressed(output / 'clip_features.npz', features=features,
                        image_embeddings=raw_image, text_embeddings=raw_text)
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return features, token_counts


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def comparison_rows(result):
    fit = result['reference_fit']
    expert = result['expert']
    return [{'photographer': expert, 'title': f"Expert {expert} | input: {prediction['variant']}",
             'target_label': 'GT: expert ISP fit', 'render_label': 'SVR prediction',
             'input_text': prediction['text'], 'target_controls': fit['normalized_controls'],
             'fit': {**fit, 'display_path': fit['fit_display_path'],
                     'fit_display_path': prediction['render'],
                     'normalized_controls': prediction['normalized_controls']},
             'descriptors': {'fit_based': True}}
            for prediction in result['predictions']]


def render_comparison(folder, result):
    draw_comparison(folder / 'comparison.png', result['image_id'], comparison_rows(result),
        heading=f"CLIP + SVR | {result['image_id']} | selected expert {result['expert']}",
        subheading='Scene-held-out predictions | Target: saved expert ISP fit | '
                   'Exact input text | GT and predicted P90 controls')


def refresh_figures(output):
    paths = sorted((output / 'images').glob('*/predictions.json'))
    if not paths:
        raise FileNotFoundError(f'No saved predictions under {output}')
    for path in paths:
        render_comparison(path.parent, json.loads(path.read_text()))
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['comparison_report'] = {
        'target': 'Saved ISP fit for the selected expert, not Expert TIFF',
        'controls': 'Ground-truth fitted and predicted six P90-normalized controls',
        'descriptor': 'Exact saved text used for each prediction variant',
        'figures': len(paths), 'renderer_sha256': sha256(Path(__file__)),
        'layout_sha256': sha256(Path(__file__).with_name('fivek_screen_comparisons.py'))}
    write_json(manifest_path, manifest)
    print(f'Refreshed {len(paths)} comparison figures; no model training or prediction changes')


def run(args):
    index = json.loads((args.source / 'index.json').read_text())
    document = load_fit_visual_descriptions()
    names, comparisons, reference_note = load_fit_references()
    selection = select_experts([entry['image_id'] for entry in index['images']], args.seed)
    if args.limit:
        selection = {image_id: selection[image_id] for image_id in sorted(selection)[:args.limit]}
    if len(selection) < args.folds:
        raise ValueError('Need at least one scene per fold')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    records = []
    config_path = None
    config_hash = None
    for entry in index['images']:
        image_id = entry['image_id']
        if image_id not in selection:
            continue
        expert = selection[image_id]
        source_path = Path(entry['results'])
        source = json.loads(source_path.read_text())
        if config_path is None:
            config_path = Path(source['calibration_config'])
            config_hash = source['calibration_config_sha256']
        if source['calibration_config_sha256'] != config_hash:
            raise ValueError('Cannot mix fit calibration configurations')
        fit = source['per_image'][image_id][expert]
        groups = document['images'][image_id][expert]
        refs = comparisons.get(image_id, {}).get(expert, {}).get('filter_references', [])
        inputs = descriptor_inputs(groups, names.get(image_id, {}).get(expert, ''), refs)
        records.append({'image_id': image_id, 'expert': expert, 'fit': fit,
                        'groups': groups, 'references': refs, 'inputs': inputs,
                        'source_result': str(source_path), 'source_sha256': sha256(source_path)})
    if sha256(config_path) != config_hash:
        raise ValueError('Source fit calibration hash no longer matches its file')
    config = json.loads(config_path.read_text())
    endpoints = pole_endpoints(config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir()
    write_json(args.out / 'selected_pairs.json', records)
    torch.set_num_threads(2)
    features, token_counts = encode_records(records, args.out, args.device)
    targets = np.asarray([[record['fit']['normalized_controls'][control] for control in CONTROL_NAMES]
                          for record in records for variant in VARIANTS])
    groups = np.asarray([record['image_id'] for record in records for variant in VARIANTS])
    predictions = np.full_like(targets, np.nan)
    fold_ids = np.full(len(targets), -1)
    baseline = np.zeros_like(targets)
    folds = []
    for fold_id, (training, testing) in enumerate(scene_folds(groups, args.folds, args.seed)):
        model = make_regressor().fit(features[training], targets[training])
        predictions[testing] = np.clip(model.predict(features[testing]), -1, 1)
        baseline[testing] = np.median(targets[training], axis=0)
        fold_ids[testing] = fold_id
        joblib.dump(model, args.out / 'models' / f'fold_{fold_id}.joblib')
        folds.append({'fold': fold_id, 'train_scenes': sorted(set(groups[training])),
                      'test_scenes': sorted(set(groups[testing]))})
        print(f'Fold {fold_id}: {len(set(groups[training]))} train scenes, '
              f'{len(set(groups[testing]))} test scenes', flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError('Missing out-of-fold predictions')
    joblib.dump(make_regressor().fit(features, targets), args.out / 'models' / 'all_data.joblib')
    write_json(args.out / 'folds.json', folds)
    metrics = {}
    for variant_index, variant in enumerate(VARIANTS):
        chosen = np.arange(variant_index, len(targets), len(VARIANTS))
        metrics[variant] = {'p90_mae': float(np.abs(predictions[chosen] - targets[chosen]).mean()),
                            'training_median_baseline_p90_mae': float(np.abs(baseline[chosen] - targets[chosen]).mean()),
                            'per_control_mae': dict(zip(CONTROL_NAMES,
                                np.abs(predictions[chosen] - targets[chosen]).mean(axis=0).tolist()))}
    csv_rows, pages = [], []
    for scene_index, record in enumerate(records):
        image_id, expert, fit = record['image_id'], record['expert'], record['fit']
        folder = args.out / 'images' / image_id
        folder.mkdir(parents=True)
        results = []
        neutral = load_linear_png(fit['neutral_source']).to(args.device)
        with Image.open(fit['fit_display_path']) as target_image:
            fit_pixels = np.asarray(target_image.convert('RGB'), dtype=np.float32) / 255
        with Image.open(fit['display_path']) as target_image:
            expert_pixels = np.asarray(target_image.convert('RGB'), dtype=np.float32) / 255
        for variant_index, variant in enumerate(VARIANTS):
            row_index = scene_index * len(VARIANTS) + variant_index
            normalized = torch.as_tensor(predictions[row_index], device=args.device, dtype=torch.float32)
            physical = normalized_to_physical(normalized, endpoints)
            with torch.inference_mode():
                rendered = render_linear(neutral, parameters_from_controls(physical, config))
                display = linear_to_srgb(rendered.clamp(0, 1))[0].permute(1, 2, 0).cpu().numpy()
            render_path = folder / f'{variant}.png'
            write_render(render_path, rendered)
            prediction = {'variant': variant, 'text': record['inputs'][variant],
                'token_count_before_truncation': token_counts[row_index],
                'clip_text_truncated': token_counts[row_index] > 75,
                'fold': int(fold_ids[row_index]),
                'normalized_controls': dict(zip(CONTROL_NAMES, predictions[row_index].tolist())),
                'physical_controls': dict(zip(CONTROL_NAMES, physical.cpu().tolist())),
                'target_normalized_controls': fit['normalized_controls'],
                'p90_mae': float(np.abs(predictions[row_index] - targets[row_index]).mean()),
                'render_mae_vs_oracle_fit': float(np.abs(display - fit_pixels).mean()),
                'render_mae_vs_expert': float(np.abs(display - expert_pixels).mean()),
                'render': str(render_path)}
            results.append(prediction)
            csv_rows.append({'image_id': image_id, 'expert': expert, 'variant': variant,
                             **{key: prediction[key] for key in ('fold', 'p90_mae', 'render_mae_vs_oracle_fit',
                                                                 'render_mae_vs_expert', 'clip_text_truncated')},
                             **{f'{name}_predicted': value for name, value in prediction['normalized_controls'].items()},
                             **{f'{name}_target': value for name, value in fit['normalized_controls'].items()},
                             'text': prediction['text']})
        result = {'image_id': image_id, 'expert': expert,
                  'reference_fit': fit, 'predictions': results}
        render_comparison(folder, result)
        write_json(folder / 'predictions.json', result)
        pages.append(f'<h2>{html.escape(image_id)} / expert {expert}</h2>'
                     f'<a href="images/{image_id}/predictions.json">Prediction data</a><br>'
                     f'<img style="max-width:100%" src="images/{image_id}/comparison.png">')
        print(f'Rendered {image_id} / expert {expert}', flush=True)
    with (args.out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    for variant in VARIANTS:
        subset = [row for row in csv_rows if row['variant'] == variant]
        for key in ('render_mae_vs_oracle_fit', 'render_mae_vs_expert'):
            metrics[variant][key] = float(np.mean([row[key] for row in subset]))
    manifest = {'method': 'Frozen OpenAI CLIP ViT-B/32 + six RBF SVRs',
        'source': str(args.source), 'annotation_sha256': sha256(FIT_VISUAL_DESCRIPTIONS),
        'source_code_sha256': sha256(Path(__file__)), 'calibration_sha256': config_hash,
        'image_input': 'Default ISP display render only; no expert or fitted image embedding',
        'text_input': 'D1-D4 separately and concatenated; combined adds existing fit style/app references when available',
        'normalization': 'L2-normalize image and text separately; concatenate; no StandardScaler',
        'normalization_reference': 'https://github.com/openai/CLIP#zero-shot-prediction',
        'selection': 'Seeded shuffled sorted scenes, balanced round-robin expert A-E; one expert per scene',
        'seed': args.seed, 'scenes': len(records), 'variants': list(VARIANTS),
        'svr': {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.05, 'gamma': 'scale',
                'hyperparameters': 'Fixed before evaluation; not selected on held-out targets'},
        'evaluation': f'{args.folds}-fold scene-disjoint out-of-fold predictions; all variants of a scene stay together',
        'all_data_model': 'For future inference only; not used for reported predictions',
        'truncated_text_inputs': sum(count > 75 for count in token_counts),
        'limitations': ['Assistant-authored descriptions are not independent human instructions.',
                       'CLIP native preprocessing resizes and center-crops Default ISP images.',
                       'Combined input may be truncated at the CLIP 77-token context.',
                       'App references are approximate and available only for a subset.'],
        'metrics': metrics}
    write_json(args.out / 'manifest.json', manifest)
    (args.out / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>FiveK CLIP + SVR</title><h1>Scene-held-out CLIP + SVR ISP predictions</h1>'
        '<p><a href="manifest.json">Protocol and metrics</a> | '
        '<a href="predictions.csv">Predicted and target controls</a></p>' + '\n'.join(pages))
    print(json.dumps({'output': str(args.out), 'scenes': len(records), 'metrics': metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--refresh-figures-only', action='store_true')
    args = parser.parse_args()
    if args.folds < 2 or args.limit < 0:
        parser.error('Require at least two folds and a nonnegative scene limit')
    if args.refresh_figures_only:
        refresh_figures(args.out)
    else:
        run(args)


if __name__ == '__main__':
    main()
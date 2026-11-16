"""Test whether D1 CLIP features predict the brightness control alone."""

import argparse
import csv
import html
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import torch

from fivek_clip_svr import VARIANTS, scene_folds, write_json
from fivek_preview import load_linear_png, write_render
from fivek_screen_comparisons import draw_comparison
from fivek_semantic_descriptions import sha256
from moran_semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints


DEFAULT_BASE = Path('/data/data1/FiveK/C_data/results/isp_match_105_clip_svr_v1')
DEFAULT_OUT = Path('/data/data1/FiveK/C_data/results/isp_match_105_d1_brightness_svr_v1')
METHODS = ('text_linear', 'text_rbf', 'image_text_linear', 'image_text_rbf', 'median_baseline')


def brightness_controls(value):
    return isolated_controls('tone_gain', value)


def isolated_controls(control, value):
    if control not in CONTROL_NAMES:
        raise ValueError(f'Unknown control: {control}')
    controls = dict.fromkeys(CONTROL_NAMES, 0.0)
    controls[control] = float(value)
    return controls


def training_search(features, targets, groups, kernel, seed):
    parameters = {'C': [0.1, 1.0, 10.0], 'epsilon': [0.02, 0.1]}
    if kernel == 'rbf':
        parameters['gamma'] = ['scale', 0.5, 2.0]
    search = GridSearchCV(SVR(kernel=kernel), parameters,
        scoring='neg_mean_absolute_error', cv=list(scene_folds(groups, 3, seed)), n_jobs=1)
    return search.fit(features, targets)


def trend_metrics(target, predicted):
    correlation = (float(np.corrcoef(target, predicted)[0, 1])
                   if np.std(target) > 1e-12 and np.std(predicted) > 1e-12 else None)
    directional = np.abs(target) >= 0.1
    return {'mae': float(mean_absolute_error(target, predicted)),
            'r2': float(r2_score(target, predicted)), 'pearson_r': correlation,
            'direction_samples': int(directional.sum()),
            'direction_accuracy': float(np.mean(np.sign(target[directional]) == np.sign(predicted[directional])))
                                  if directional.any() else None}


def run(args):
    control = getattr(args, 'control', 'tone_gain')
    methods = ('text_linear', 'median_baseline') if getattr(args, 'text_linear_only', False) else METHODS
    label = 'brightness' if control == 'tone_gain' else control
    records = json.loads((args.base / 'selected_pairs.json').read_text())
    source_manifest = json.loads((args.base / 'manifest.json').read_text())
    if source_manifest['variants'] != list(VARIANTS):
        raise ValueError('Cached feature variants differ from the expected order')
    groups = np.asarray([record['image_id'] for record in records])
    if len(np.unique(groups)) != len(groups) or len(groups) < 6:
        raise ValueError('Require at least six unique scenes')
    with np.load(args.base / 'clip_features.npz') as cache:
        features = cache['features'][::len(VARIANTS)]
    if len(features) != len(records) or features.shape[1] % 2:
        raise ValueError('Invalid cached D1 feature dimensions')
    midpoint = features.shape[1] // 2
    text_features = features[:, midpoint:]
    target = np.asarray([record['fit']['normalized_controls'][control] for record in records])
    predictions = {method: np.full(len(records), np.nan) for method in methods}
    fold_ids = np.full(len(records), -1)
    fold_records = []
    source_result = json.loads(Path(records[0]['source_result']).read_text())
    config_path = Path(source_result['calibration_config'])
    if sha256(config_path) != source_manifest['calibration_sha256']:
        raise ValueError('Calibration differs from the source experiment')
    config = json.loads(config_path.read_text())
    endpoints = pole_endpoints(config)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir()
    for fold, (training, testing) in enumerate(scene_folds(groups, 5, args.seed)):
        details = {'fold': fold, 'train_scenes': groups[training].tolist(),
                   'test_scenes': groups[testing].tolist(), 'selected_parameters': {}}
        predictions['median_baseline'][testing] = np.median(target[training])
        fold_ids[testing] = fold
        for method in methods[:-1]:
            inputs = text_features if method.startswith('text_') else features
            kernel = 'linear' if method.endswith('linear') else 'rbf'
            search = training_search(inputs[training], target[training], groups[training], kernel, args.seed)
            predictions[method][testing] = np.clip(search.predict(inputs[testing]), -1, 1)
            details['selected_parameters'][method] = {'parameters': search.best_params_,
                                                       'inner_validation_mae': float(-search.best_score_)}
            joblib.dump(search.best_estimator_, args.out / 'models' / f'{method}_fold_{fold}.joblib')
        fold_records.append(details)
        print(f'{control} fold {fold}: {len(training)} training / {len(testing)} held-out scenes', flush=True)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('Incomplete held-out predictions')
    metrics = {method: trend_metrics(target, values) for method, values in predictions.items()}
    figure, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), constrained_layout=True)
    for axis, method in zip(axes, methods):
        axis.scatter(target, predictions[method], s=15, alpha=0.7)
        axis.plot([-1, 1], [-1, 1], '--', color='gray')
        axis.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=f'GT {label} (P90)',
             ylabel=f'Held-out predicted {label} (P90)',
                 title=f"{method}\nMAE={metrics[method]['mae']:.3f}; R2={metrics[method]['r2']:.3f}")
        axis.set_aspect('equal')
    figure.savefig(args.out / f'{control}_trends.png', dpi=180)
    plt.close(figure)
    torch.set_num_threads(2)
    csv_rows, links = [], []
    for index, record in enumerate(records):
        folder = args.out / 'images' / record['image_id']
        folder.mkdir(parents=True)
        neutral = load_linear_png(record['fit']['neutral_source']).to(args.device)
        gt_controls = isolated_controls(control, target[index])
        target_path = folder / f'target_{control}_only.png'
        rows, scene_predictions = [], []
        for method, value in [('target', target[index]),
                              *[(method, predictions[method][index]) for method in methods]]:
            controls = isolated_controls(control, value)
            normalized = torch.tensor([controls[name] for name in CONTROL_NAMES],
                                      device=args.device, dtype=torch.float32)
            physical = normalized_to_physical(normalized, endpoints)
            with torch.inference_mode():
                output = render_linear(neutral, parameters_from_controls(physical, config))
            output_path = target_path if method == 'target' else folder / f'{method}.png'
            write_render(output_path, output)
            if method == 'target':
                continue
            predicted = {'method': method, 'text': record['inputs']['D1'],
                         'control': control, 'target_p90': float(target[index]), 'predicted_p90': float(value),
                         'predicted_physical': float(physical[CONTROL_NAMES.index(control)]),
                         'fold': int(fold_ids[index]), 'render': str(output_path)}
            scene_predictions.append(predicted)
            csv_rows.append({'image_id': record['image_id'], 'expert': record['expert'], **predicted})
            rows.append({'photographer': record['expert'], 'title': method,
                'target_label': f'GT: {label} only', 'render_label': f'Prediction: {label} only',
                'target_controls': gt_controls, 'input_text': record['inputs']['D1'],
                'descriptors': {'fit_based': True},
                'fit': {**record['fit'], 'display_path': str(target_path),
                        'fit_display_path': str(output_path), 'normalized_controls': controls}})
        draw_comparison(folder / 'comparison.png', record['image_id'], rows,
            heading=f"D1 {label}-only SVR | {record['image_id']} | expert {record['expert']}",
            subheading=f'Five-fold scene-held-out | GT and prediction change {label} only | '
                       'All other controls neutral | Exact D1 text')
        write_json(folder / 'predictions.json', {'image_id': record['image_id'], 'expert': record['expert'],
                   'reference_full_fit': record['fit'], 'target_isolated_controls': gt_controls,
                   'predictions': scene_predictions})
        links.append(f'<h2>{html.escape(record["image_id"])}</h2>'
                     f'<img style="max-width:100%" src="images/{record["image_id"]}/comparison.png">')
    with (args.out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(args.out / 'folds.json', fold_records)
    write_json(args.out / 'manifest.json', {
        'experiment': f'D1 {control}-only SVR trend study', 'control': control, 'source': str(args.base),
        'source_features_sha256': sha256(args.base / 'clip_features.npz'),
        'source_pairs_sha256': sha256(args.base / 'selected_pairs.json'),
        'source_code_sha256': sha256(Path(__file__)), 'scenes': len(records), 'seed': args.seed,
        'text_input': 'Exact selected expert fit D1 only',
        'target': f'Fitted P90-normalized {control}',
        'features': 'Cached L2-normalized CLIP text embeddings only' if args.text_linear_only else
                'Cached separately L2-normalized CLIP image/text embeddings; no new encoding',
        'evaluation': 'Five outer scene folds; three inner scene folds select C/epsilon/gamma on training scenes only',
        'rendering': f'Target and prediction isolate {control}; all five other normalized controls are zero',
        'limitations': 'Targets were originally jointly fitted with five other controls; D1 notes may omit the selected control.',
        'methods': list(methods), 'metrics': metrics})
    (args.out / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        f'<title>D1 {label} SVR</title><h1>D1 {label}-only SVR</h1>'
        '<p><a href="manifest.json">Protocol and metrics</a> | '
        '<a href="predictions.csv">Predictions</a></p>'
        f'<img style="max-width:100%" src="{control}_trends.png">' + '\n'.join(links))
    print(json.dumps({'output': str(args.out), 'metrics': metrics}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--control', choices=CONTROL_NAMES, default='tone_gain')
    parser.add_argument('--text-linear-only', action='store_true')
    parser.add_argument('--all-controls', action='store_true')
    args = parser.parse_args()
    if not args.all_controls:
        run(args)
        return
    if not args.text_linear_only:
        parser.error('--all-controls requires --text-linear-only')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    root = args.out
    summary = {}
    for control in CONTROL_NAMES:
        args.control, args.out = control, root / control
        summary[control] = run(args)
    write_json(root / 'all_controls_metrics.json', summary)
    table = ['<table border="1"><tr><th>Control</th><th>SVR MAE</th><th>Median MAE</th>'
             '<th>R-squared</th><th>Pearson r</th></tr>']
    for control, metrics in summary.items():
        model, baseline = metrics['text_linear'], metrics['median_baseline']
        table.append(f'<tr><td><a href="{control}/index.html">{control}</a></td>'
                     f'<td>{model["mae"]:.4f}</td><td>{baseline["mae"]:.4f}</td>'
                     f'<td>{model["r2"]:.4f}</td><td>{model["pearson_r"]}</td></tr>')
    table.append('</table>')
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>D1 text-only linear SVR: six controls</title>'
        '<h1>D1 text-only linear SVR: six isolated controls</h1>' + ''.join(table))


if __name__ == '__main__':
    main()
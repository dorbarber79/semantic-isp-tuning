"""Train frozen-CLIP MLPs and compare with matched text-linear SVR/median baselines."""

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
import open_clip
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch import nn
import torch.nn.functional as F

from fivek_brightness_svr import training_search
from fivek_clip_svr import fused_features, select_experts, scene_folds, write_json
from fivek_preview import load_linear_png, write_render
from fivek_screen_comparisons import draw_comparison
from fivek_semantic_descriptions import sha256
from semantic_isp import CONTROL_NAMES, linear_to_srgb, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints


DEFAULT_RELEASE = Path('/data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-14-v2')
DEFAULT_OUT = Path('/data/data1/FiveK/C_data/results/isp_match_125_mlp_d1_v1')
CONTROL_VARIANT = 'mlp_image_text'
METHODS = ('median', 'svm_text', 'mlp_text', 'mlp_image_text')


class ControlMLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, len(CONTROL_NAMES)), nn.Tanh())

    def forward(self, features):
        return self.layers(features)


def fusion_features(image, text):
    if image.shape != text.shape or image.ndim != 2:
        raise ValueError('Image and text embeddings must be matching matrices')
    width = image.shape[1]
    interactions = image * text * np.sqrt(width)
    distance = np.abs(image - text) / np.sqrt(2)
    return np.concatenate((image, text, interactions, distance), axis=1)


def choose_experts(scene_ids, original_pairs, seed=0):
    original = {row['image_id']: row['expert'] for row in original_pairs}
    scenes = set(scene_ids)
    if not set(original) <= scenes or any(expert not in 'ABCDE' for expert in original.values()):
        raise ValueError('Original selected experts do not match release scenes')
    return {**original, **select_experts(sorted(scenes - set(original)), seed)}


def split_inner_scenes(train_indices, scene_ids, seed, validation_fraction=0.15):
    training_scenes = np.unique(scene_ids[train_indices])
    rng = np.random.default_rng(seed)
    order = rng.permutation(training_scenes)
    validation_count = max(1, int(round(validation_fraction * len(order))))
    validation_scenes = set(order[:validation_count])
    fit_indices = np.asarray([i for i in train_indices if scene_ids[i] not in validation_scenes])
    validation_indices = np.asarray([i for i in train_indices if scene_ids[i] in validation_scenes])
    if not len(fit_indices) or not len(validation_indices):
        raise ValueError('Inner scene split produced an empty partition')
    return fit_indices, validation_indices


def fit_mlp(features, targets, fit_indices, validation_indices, hidden, dropout,
            seed, device, max_epochs=500, patience=45):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_features = features[fit_indices]
    mean = train_features.mean(axis=0)
    scale = np.maximum(train_features.std(axis=0), 0.05)
    normalized = np.clip((features - mean) / scale, -8, 8)
    x = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    model = ControlMLP(features.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    generator = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        model.train()
        permutation = generator.permutation(fit_indices)
        for start in range(0, len(permutation), 32):
            batch = permutation[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(x[batch]), y[batch], beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(F.smooth_l1_loss(
                model(x[validation_indices]), y[validation_indices], beta=0.1).cpu())
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError('MLP early stopping did not save a checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    checkpoint = {'state_dict': best_state, 'input_mean': mean, 'input_scale': scale,
                  'input_size': features.shape[1], 'hidden_size': hidden, 'dropout': dropout,
                  'best_epoch': best_epoch, 'validation_loss': best_loss}
    return np.clip(prediction, -1, 1), checkpoint


def score_controls(target, prediction):
    result = {}
    for index, name in enumerate(CONTROL_NAMES):
        actual, predicted = target[:, index], prediction[:, index]
        active = np.abs(actual) >= 0.1
        result[name] = {'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(np.mean((actual - predicted) ** 2))),
            'r2': float(r2_score(actual, predicted)),
            'pearson_r': float(np.corrcoef(actual, predicted)[0, 1])
                if actual.std() > 1e-12 and predicted.std() > 1e-12 else None,
            'within_0_1_percent': float(100 * np.mean(np.abs(actual - predicted) <= 0.1)),
            'target_active_count': int(active.sum()),
            'active_mae': float(np.abs(actual[active] - predicted[active]).mean()) if active.any() else None}
    result['macro_mae'] = float(np.mean([result[name]['mae'] for name in CONTROL_NAMES]))
    return result


def refresh_report(output):
    metrics = json.loads((output / 'metrics.json').read_text())
    predictions = json.loads((output / 'out_of_fold_predictions.json').read_text())
    target = np.asarray([[row['target'][name] for name in CONTROL_NAMES] for row in predictions])
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    colors = {'svm_text': '#176b70', 'mlp_text': '#b8593d',
              'mlp_image_text': '#62549a', 'median': '#728292'}
    for column, (axis, control) in enumerate(zip(axes.flat, CONTROL_NAMES)):
        for method in ('median', 'svm_text', 'mlp_text', 'mlp_image_text'):
            values = np.asarray([row['predictions'][method][control] for row in predictions])
            axis.scatter(target[:, column], values, s=12, alpha=.42,
                         color=colors[method], label=method)
        axis.plot([-1, 1], [-1, 1], color='#333333', linestyle='--', linewidth=1)
        axis.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='GT P90 control',
                 ylabel='Scene-held-out prediction', title=control)
        axis.set_aspect('equal')
        if column == 0:
            axis.legend(fontsize=8)
    figure.savefig(output / 'out_of_fold_by_control.png', dpi=180)
    plt.close(figure)
    rows = []
    for method in ('median', 'svm_text', 'mlp_text', 'mlp_image_text'):
        rows.append('<tr><th>' + html.escape(method) + '</th><td>'
            + f"{metrics[method]['macro_mae']:.4f}</td>"
            + ''.join(f"<td>{metrics[method][control]['mae']:.4f}</td>" for control in CONTROL_NAMES)
            + '</tr>')
    intervals = json.loads((output / 'summary.json').read_text())['bootstrap_mae_difference_vs_svm']
    ci_rows = ''.join(f'<li>{html.escape(method)} minus SVM macro-MAE difference: '
                      f'95% paired scene-bootstrap interval '
                      f'[{float(bounds[0]):.4f}, {float(bounds[1]):.4f}]</li>'
                      for method, bounds in intervals.items())
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>FiveK frozen-CLIP MLP</title>'
        '<h1>Frozen-CLIP MLP versus SVM and median</h1>'
        '<p>125 selected scenes; one expert fit per scene; five scene-disjoint folds. '
        'These are parameter-supervised experiments; the MLP has no render-level loss.</p>'
        '<p><a href="summary.json">Training protocol</a> | '
        '<a href="metrics.json">Metrics</a> | '
        '<a href="out_of_fold_predictions.csv">Out-of-fold predictions</a></p>'
        '<table border="1"><thead><tr><th>Method</th><th>Macro MAE</th>'
        + ''.join(f'<th>{html.escape(control)}</th>' for control in CONTROL_NAMES)
        + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
        '<h2>Paired scene-bootstrap differences from SVM</h2><ul>' + ci_rows + '</ul>'
        '<h2>Out-of-fold predictions by control</h2>'
        '<img style="max-width:100%" src="out_of_fold_by_control.png">')


def image_scores(target, prediction):
    target_srgb = np.clip(linear_to_srgb(target.clamp(0, 1)), 0, 1)
    prediction_srgb = np.clip(linear_to_srgb(prediction.clamp(0, 1)), 0, 1)
    target_np = target_srgb[0].permute(1, 2, 0).cpu().numpy()
    prediction_np = prediction_srgb[0].permute(1, 2, 0).cpu().numpy()
    mae = float(np.abs(target_np - prediction_np).mean() * 255)
    mse = float(np.square(target_np - prediction_np).mean())
    return {'mae_255': mae, 'psnr_db': float(10 * np.log10(1 / mse)) if mse > 0 else None}


def run(args):
    release = json.loads((args.release / 'dataset_manifest.json').read_text())
    pair_rows = json.loads((args.release / 'annotations/normalized_pairs.json').read_text())
    scene_ids = np.asarray(sorted({row['image_id'] for row in pair_rows}))
    original_pairs = json.loads((Path('/data/data1/FiveK/C_data/results/isp_match_105_clip_svr_v1')
                                 / 'selected_pairs.json').read_text())
    expert_by_scene = choose_experts(scene_ids.tolist(), original_pairs, args.seed)
    pair_map = {(row['image_id'], row['expert']): row for row in pair_rows}
    selected = [pair_map[image_id, expert_by_scene[image_id]] for image_id in scene_ids]
    records, texts, default_paths, target_paths = [], [], [], []
    for row in selected:
        fit_layer = next(layer for layer in row['annotation_layers']
                         if layer['annotation_target'] == 'fitted_output_vs_default')
        groups = fit_layer['value']
        if not groups.get('D1'):
            raise ValueError(f'Missing D1 annotation: {row["image_id"]}/{row["expert"]}')
        default_path = args.release / row['assets']['default_display_720']['path']
        target_rel = next(asset['path'] for asset in row['assets'].values()
                          if asset['path'].endswith(f'experts/{row["expert"]}/fit_720.png'))
        expert_rel = next(asset['path'] for asset in row['assets'].values()
                          if asset['path'].endswith(f'experts/{row["expert"]}/display_720.png'))
        texts.append(groups['D1'])
        default_paths.append(default_path)
        target_paths.append(args.release / target_rel)
        records.append({'image_id': row['image_id'], 'expert': row['expert'],
                        'normalized_controls': row['normalized_controls'],
                        'physical_controls': row['physical_controls'], 'D1': groups['D1'],
                        'default_path': str(default_path), 'target_path': str(args.release / target_rel),
                        'expert_path': str(args.release / expert_rel),
                        'calibration': row['calibration'], 'fit_metadata': row['fit_metadata']})
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    clip, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True, device=args.device)
    clip.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    image_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(records), 16):
            images = []
            for path in default_paths[start:start + 16]:
                with Image.open(path) as image:
                    images.append(preprocess(image.convert('RGB')))
            image_features = clip.encode_image(torch.stack(images).to(args.device)).float()
            image_embeddings.extend(F.normalize(image_features, dim=-1).cpu().numpy())
        text_embeddings = []
        for start in range(0, len(texts), 64):
            text_features = clip.encode_text(tokenizer(texts[start:start + 64]).to(args.device)).float()
            text_embeddings.extend(F.normalize(text_features, dim=-1).cpu().numpy())
    image_embeddings, text_embeddings = np.asarray(image_embeddings), np.asarray(text_embeddings)
    np.savez_compressed(args.out / 'clip_embeddings.npz', image=image_embeddings, text=text_embeddings)
    del clip
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    d1_svm_report = Path('/data/data1/FiveK/C_data/results/isp_match_105_screen_d1_text_linear_v1')
    old_svm = {}
    for path in (d1_svm_report / 'images').glob('*/predictions.json'):
        result = json.loads(path.read_text())
        for row in result['rows']:
            if row['previous_selected_pair']:
                old_svm[row['image_id']] = [row['normalized_controls'][name] for name in CONTROL_NAMES]
    if not set(old_svm) <= set(scene_ids):
        raise ValueError('Existing SVM subset contains scenes outside release')
    original_fold_manifest = Path('/data/data1/FiveK/C_data/results/isp_match_105_d1_text_linear_all_controls_v1')
    same_fold_groups = json.loads((original_fold_manifest / 'warmth/folds.json').read_text())
    records_by_id = {row['image_id']: row for row in records}
    selected_ids = set(old_svm)
    if any(records_by_id[image_id]['expert'] != expert_by_scene[image_id] for image_id in selected_ids):
        raise ValueError('Selected experts changed for the original SVM evaluation scenes')
    folds = list(scene_folds(scene_ids, 5, args.seed))
    if set(scene_ids) == {scene for fold in same_fold_groups for scene in fold['train_scenes'] + fold['test_scenes']}:
        expected = [(fold['train_scenes'], fold['test_scenes']) for fold in same_fold_groups]
        actual = [(scene_ids[train].tolist(), scene_ids[test].tolist()) for train, test in folds]
        if actual != expected:
            raise ValueError('Fold definitions do not reproduce the saved SVM test split')
    targets = np.asarray([[row['normalized_controls'][name] for name in CONTROL_NAMES] for row in records], dtype=np.float32)
    text_inputs, fusion_inputs = text_embeddings, fusion_features(image_embeddings, text_embeddings)
    predictions = {method: np.full_like(targets, np.nan, dtype=np.float64)
                   for method in ('median', 'svm_text', 'mlp_text', 'mlp_image_text')}
    fold_rows, fold_ids = [], np.full(len(records), -1)
    for fold_id, (training, testing) in enumerate(folds):
        groups = scene_ids
        fit_indices, validation_indices = split_inner_scenes(training, groups, args.seed + fold_id)
        predictions['median'][testing] = np.median(targets[training], axis=0)
        for method, matrix in (('mlp_text', text_inputs), ('mlp_image_text', fusion_inputs)):
            hidden = 16 if method == 'mlp_text' else 8
            pred, checkpoint = fit_mlp(matrix, targets, fit_indices, validation_indices,
                                       hidden, 0.3, args.seed + 100 * fold_id + hidden,
                                       args.device)
            predictions[method][testing] = pred[testing]
            torch.save(checkpoint, args.out / 'models' / f'{method}_fold_{fold_id}.pt')
            fold_rows.append({'fold': fold_id, 'method': method,
                'train_scenes': scene_ids[fit_indices].tolist(),
                'validation_scenes': scene_ids[validation_indices].tolist(),
                'test_scenes': scene_ids[testing].tolist(), 'hidden_units': hidden,
                'dropout': 0.3, 'best_epoch': checkpoint['best_epoch'],
                'validation_loss': checkpoint['validation_loss']})
        svm_predictions = np.full((len(records), len(CONTROL_NAMES)), np.nan)
        for column, control in enumerate(CONTROL_NAMES):
            model = training_search(text_inputs[training], targets[training, column],
                                    scene_ids[training], 'linear', args.seed)
            svm_predictions[testing, column] = np.clip(model.predict(text_inputs[testing]), -1, 1)
            joblib.dump(model.best_estimator_, args.out / 'models' / f'svm_text_{control}_fold_{fold_id}.joblib')
        predictions['svm_text'][testing] = svm_predictions[testing]
        fold_ids[testing] = fold_id
        print(f'Fold {fold_id}: {len(fit_indices)} train, {len(validation_indices)} validation, '
              f'{len(testing)} test scenes', flush=True)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('Some out-of-fold predictions are missing')
    metrics = {method: score_controls(targets, pred) for method, pred in predictions.items()}
    old_svm_subset = np.asarray([[records_by_id[key]['normalized_controls'][name]
                                  for name in CONTROL_NAMES] for key in old_svm], dtype=np.float64)
    old_svm_pred = np.asarray([old_svm[key] for key in old_svm], dtype=np.float64)
    if old_svm_subset.shape != old_svm_pred.shape or old_svm_subset.shape[1:] != (len(CONTROL_NAMES),):
        raise ValueError('Historical SVM predictions do not match six-control target shape')
    metrics['svm_text']['original_105_pairs'] = score_controls(old_svm_subset, old_svm_pred)
    deltas = {method: predictions[method] - predictions['svm_text']
              for method in ('mlp_text', 'mlp_image_text')}
    bootstrap = {}
    rng = np.random.default_rng(args.seed)
    for method, delta in deltas.items():
        scene_mae_delta = np.abs(predictions[method] - targets).mean(axis=1) - \
                          np.abs(predictions['svm_text'] - targets).mean(axis=1)
        samples = scene_mae_delta[rng.integers(0, len(scene_mae_delta), (2000, len(scene_mae_delta)))].mean(axis=1)
        bootstrap[method] = np.quantile(samples, [.025, .975]).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({'text_mean': text_inputs.mean(axis=0), 'text_std': text_inputs.std(axis=0),
                'fusion_mean': fusion_inputs.mean(axis=0), 'fusion_std': fusion_inputs.std(axis=0)},
               args.out / 'feature_statistics.pt')
    write_json(args.out / 'metrics.json', metrics)
    write_json(args.out / 'folds.json', fold_rows)
    write_json(args.out / 'out_of_fold_predictions.json', [
        {'image_id': records[i]['image_id'], 'expert': records[i]['expert'], 'D1': records[i]['D1'],
         'fold': int(fold_ids[i]), 'target': records[i]['normalized_controls'],
         'predictions': {method: dict(zip(CONTROL_NAMES, values[i].tolist()))
                         for method, values in predictions.items()}}
        for i in range(len(records))])
    csv_rows = []
    for i, record in enumerate(records):
        csv_rows.append({'image_id': record['image_id'], 'expert': record['expert'], 'fold': int(fold_ids[i]),
            'D1': record['D1'], **{f'{name}_target': targets[i, j] for j, name in enumerate(CONTROL_NAMES)},
            **{f'{name}_{method}': predictions[method][i, j] for j, name in enumerate(CONTROL_NAMES)
               for method in predictions}})
    with (args.out / 'out_of_fold_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    summary = {'experiment': 'Frozen CLIP image/text + small supervised MLP for six P90 controls',
        'release': str(args.release), 'scenes': len(records), 'scene_folds': 5,
        'selected_experts': {row['image_id']: row['expert'] for row in records},
        'text': 'Exact D1; assistant-authored visual-fit description, not a user request',
        'clip': 'OpenAI ViT-B/32 frozen; separately L2-normalized image/text embeddings',
        'mlp_text': {'hidden': 16, 'dropout': 0.3, 'loss': 'SmoothL1 beta=0.1',
                     'optimizer': 'AdamW lr=0.001 weight_decay=0.01', 'inner_validation_scene_fraction': 0.15,
                     'early_stopping_patience': 45},
        'mlp_image_text': {'input': '[image,text,image*text*sqrt(512),abs(image-text)/sqrt(2)]',
                           'hidden': 8, 'dropout': 0.3, 'loss': 'SmoothL1 beta=0.1',
                           'optimizer': 'AdamW lr=0.001 weight_decay=0.01'},
        'baselines': {'median': 'Training-fold componentwise median',
                      'svm_text': 'Independent text-only linear SVRs, same outer folds; original 105-pair result also reported'},
        'fold_protocol': 'Outer folds keep all five expert fits of each scene together; MLP early stopping uses validation scenes inside each training fold; SVR hyperparameters selected by inner scene CV.',
        'bootstrap_mae_difference_vs_svm': bootstrap,
        'metrics': metrics,
        'limitations': ['Only 125 independent scenes; all five fits share the same scene.',
                        'Descriptions describe fitted appearance, not independently collected user requests.',
                        'No render-level loss or human evaluation was used in MLP training.',
                        'MLP fit is an exploratory benchmark and does not establish improved generalization.']}
    write_json(args.out / 'summary.json', summary)
    refresh_report(args.out)
    print(json.dumps({method: value['macro_mae'] for method, value in metrics.items()}, indent=2))
    print('output', args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--refresh-report-only', action='store_true')
    args = parser.parse_args()
    if args.refresh_report_only:
        refresh_report(args.out)
    else:
        run(args)


if __name__ == '__main__':
    main()
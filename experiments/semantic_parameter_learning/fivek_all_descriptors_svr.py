"""Compare D1 with exact D1-D4 text using independent linear SVRs."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from fivek_brightness_svr import training_search, trend_metrics
from fivek_clip_svr import write_json
from fivek_screen_comparisons import FIT_VISUAL_DESCRIPTIONS, load_fit_visual_descriptions
from fivek_semantic_descriptions import GROUPS, sha256
from moran_semantic_isp import CONTROL_NAMES


DEFAULT_REPORT = Path('/data/data1/FiveK/C_data/results/isp_match_105_screen_d1_text_linear_v1')
DEFAULT_MODELS = Path('/data/data1/FiveK/C_data/results/isp_match_105_d1_text_linear_all_controls_v1')
DEFAULT_OUT = Path('/data/data1/FiveK/C_data/results/isp_match_105_d1_vs_all_d_linear_v1')


def all_descriptor_text(groups):
    if set(groups) != set(GROUPS):
        raise ValueError('Require exactly D1-D4')
    return ' '.join(groups[group] for group in GROUPS)


def comparison_metrics(target, d1, all_d, baseline):
    result = {name: trend_metrics(target, prediction)
              for name, prediction in [('D1', d1), ('D1_D4', all_d), ('median', baseline)]}
    for name, prediction in [('D1', d1), ('D1_D4', all_d), ('median', baseline)]:
        result[name]['within_0_1_percent'] = float(100 * np.mean(np.abs(target - prediction) <= 0.1))
    result['all_d_relative_mae_improvement_percent'] = float(
        100 * (result['D1']['mae'] - result['D1_D4']['mae']) / max(result['D1']['mae'], 1e-12))
    return result


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    descriptions = load_fit_visual_descriptions()
    records = []
    for path in sorted((args.report / 'images').glob('*/predictions.json')):
        scene = json.loads(path.read_text())
        if len(scene['rows']) != 5 or {row['expert'] for row in scene['rows']} != set('ABCDE'):
            raise ValueError('Require all five expert rows per scene')
        for row in sorted(scene['rows'], key=lambda value: value['expert']):
            groups = descriptions['images'][row['image_id']][row['expert']]
            if groups['D1'] != row['text']:
                raise ValueError('D1 source changed since the comparison report')
            records.append({'image_id': row['image_id'], 'expert': row['expert'],
                            'text_D1': groups['D1'], 'text_all_D': all_descriptor_text(groups),
                            'selected': row['previous_selected_pair'], 'fold': row['fold'],
                            'target': row['target_normalized_controls'],
                            'D1_prediction': row['normalized_controls']})
    if not records:
        raise FileNotFoundError('No source report records')
    scenes = np.asarray([record['image_id'] for record in records])
    selected = np.asarray([record['selected'] for record in records], dtype=bool)
    target = np.asarray([[record['target'][name] for name in CONTROL_NAMES] for record in records])
    d1 = np.asarray([[record['D1_prediction'][name] for name in CONTROL_NAMES] for record in records])
    all_d = np.full_like(target, np.nan)
    baseline = np.full_like(target, np.nan)
    folds = json.loads((args.models / 'warmth/folds.json').read_text())
    for control in CONTROL_NAMES:
        current = json.loads((args.models / control / 'folds.json').read_text())
        if [(fold['train_scenes'], fold['test_scenes']) for fold in current] != [
                (fold['train_scenes'], fold['test_scenes']) for fold in folds]:
            raise ValueError('D1 control models use different folds')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir()
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True, device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    texts = [record['text_all_D'] for record in records]
    token_counts = [len(tokenizer.encode(text)) for text in texts]
    if max(token_counts) > 75:
        raise ValueError('Combined descriptors exceed CLIP context; require an explicit pooling protocol')
    features = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            embedding = model.encode_text(tokenizer(texts[start:start + 64]).to(args.device)).float()
            features.extend(F.normalize(embedding, dim=-1).cpu().numpy())
    features = np.asarray(features)
    np.savez_compressed(args.out / 'all_D_text_embeddings.npz', features=features)
    del model
    fold_details = []
    for fold in folds:
        if set(fold['train_scenes']) & set(fold['test_scenes']):
            raise ValueError('Outer fold leaks scenes')
        training = np.flatnonzero(np.isin(scenes, fold['train_scenes']) & selected)
        testing = np.flatnonzero(np.isin(scenes, fold['test_scenes']))
        if len(training) != len(fold['train_scenes']):
            raise ValueError('Training expert selection differs from D1 experiment')
        if any(records[index]['fold'] != fold['fold'] for index in testing):
            raise ValueError('Reported D1 prediction uses a different held-out fold')
        detail = {'fold': fold['fold'], 'train_scenes': fold['train_scenes'],
                  'test_scenes': fold['test_scenes'], 'parameters': {}}
        for column, control in enumerate(CONTROL_NAMES):
            search = training_search(features[training], target[training, column],
                                     scenes[training], 'linear', args.seed)
            all_d[testing, column] = np.clip(search.predict(features[testing]), -1, 1)
            baseline[testing, column] = np.median(target[training, column])
            joblib.dump(search.best_estimator_, args.out / 'models' / f'{control}_fold_{fold["fold"]}.joblib')
            detail['parameters'][control] = search.best_params_
        fold_details.append(detail)
        print(f'Completed all-D linear fold {fold["fold"]}', flush=True)
    if not np.isfinite(all_d).all():
        raise ValueError('Incomplete held-out predictions')
    summaries = {}
    for label, mask in [('all_525_pairs', np.ones(len(records), dtype=bool)),
                        ('selected_105_pairs', selected)]:
        summaries[label] = {control: comparison_metrics(target[mask, column], d1[mask, column],
                                all_d[mask, column], baseline[mask, column])
                            for column, control in enumerate(CONTROL_NAMES)}
    paired_delta = (np.abs(all_d - target) - np.abs(d1 - target)).reshape(-1, 5, 6).mean(axis=1)
    rng = np.random.default_rng(args.seed)
    bootstrap = paired_delta[rng.integers(0, len(paired_delta), size=(2000, len(paired_delta)))].mean(axis=1)
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    for column, control in enumerate(CONTROL_NAMES):
        summaries['all_525_pairs'][control]['all_d_minus_d1_mae_scene_bootstrap_95_interval'] = intervals[:, column].tolist()
    csv_rows = []
    for index, record in enumerate(records):
        record['all_D_prediction'] = dict(zip(CONTROL_NAMES, all_d[index].tolist()))
        row = {'image_id': record['image_id'], 'expert': record['expert'],
               'fold': record['fold'], 'selected_training_pair': record['selected'],
               'text_D1': record['text_D1'], 'text_all_D': record['text_all_D']}
        for column, control in enumerate(CONTROL_NAMES):
            row.update({f'{control}_target': target[index, column],
                        f'{control}_D1': d1[index, column], f'{control}_all_D': all_d[index, column],
                        f'{control}_median': baseline[index, column]})
        csv_rows.append(row)
    with (args.out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(args.out / 'predictions.json', records)
    write_json(args.out / 'folds.json', fold_details)
    write_json(args.out / 'metrics.json', summaries)
    write_json(args.out / 'manifest.json', {
        'experiment': 'D1 versus exact concatenation of D1-D4; independent text-only linear SVRs',
        'source_report': str(args.report), 'source_models': str(args.models),
        'source_predictions_sha256': sha256(args.report / 'predictions.csv'),
        'descriptions_sha256': sha256(FIT_VISUAL_DESCRIPTIONS), 'source_code_sha256': sha256(Path(__file__)),
        'scenes': len(set(scenes)), 'pairs': len(records), 'selected_pairs': int(selected.sum()),
        'seed': args.seed, 'max_text_tokens': max(token_counts), 'truncated_inputs': 0,
        'training': 'Same selected expert per scene as D1; five identical outer folds; same three-fold inner C/epsilon search',
        'features': 'Frozen OpenAI CLIP ViT-B/32, separate L2 text normalization, no image features',
        'text': 'D1 D2 D3 D4 concatenated verbatim; no filter names or app references',
        'bootstrap': '2000 paired scene-level resamples of saved OOF error differences; no model refitting, no multiple-comparison adjustment'})
    print(json.dumps({'output': str(args.out), 'all_pairs_metrics': summaries['all_525_pairs']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
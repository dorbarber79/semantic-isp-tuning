"""Summarize ISP fits after explicitly trimming low-PSNR expert-fit pairs."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fivek_clip_svr import write_json
from fivek_semantic_descriptions import sha256
from semantic_isp import CONTROL_NAMES
from package_fivek_dataset import control_statistics


DEFAULT_DATASET = Path('/data/data1/FiveK/datasets/FiveK-ISPVA')
DEFAULT_OUTPUT = DEFAULT_DATASET / 'analyses/fit_psnr_trimmed_5pct_v1'
TRIM_FRACTIONS = (0.05, 0.10)


def load_psnr_pairs(release):
    rows = json.loads((release / 'annotations/normalized_pairs.json').read_text())
    scored = []
    for row in rows:
        target_path = release / row['assets']['expert_display_720']['path']
        fit_path = release / row['assets']['fit_display_720']['path']
        with Image.open(target_path) as image:
            target = np.asarray(image.convert('RGB'), dtype=np.float64)
        with Image.open(fit_path) as image:
            fit = np.asarray(image.convert('RGB'), dtype=np.float64)
        if target.shape != fit.shape:
            raise ValueError(f'Expert/fit dimensions differ: {row["image_id"]}/{row["expert"]}')
        difference = fit - target
        mse = float(np.mean(difference ** 2))
        score = math.inf if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
        scored.append({**row, 'psnr_db': score, 'pixel_mse_255': mse,
                       'pixel_mae_255': float(np.abs(difference).mean()),
                       'source_result': row['fit_metadata']})
    return scored


def trim_lowest(rows, fraction):
    if not 0 <= fraction < 1:
        raise ValueError('Trim fraction must be in [0, 1)')
    count = int(math.ceil(len(rows) * fraction))
    ranked = sorted(rows, key=lambda row: (row['psnr_db'], row['image_id'], row['expert']))
    excluded = ranked[:count]
    excluded_ids = {(row['image_id'], row['expert']) for row in excluded}
    retained = [row for row in rows if (row['image_id'], row['expert']) not in excluded_ids]
    return retained, excluded


def flatten_control_rows(rows):
    flattened = []
    for row in rows:
        flattened.append({'image_id': row['image_id'], 'expert': row['expert'],
            'psnr_db': row['psnr_db'], 'pixel_mse_255': row['pixel_mse_255'],
            'pixel_mae_255': row['pixel_mae_255'],
            **{f'{name}_p90': row['normalized_controls'][name] for name in CONTROL_NAMES},
            **{f'{name}_physical': row['physical_controls'][name] for name in CONTROL_NAMES}})
    return flattened


def summarise(rows):
    scores = np.asarray([row['psnr_db'] for row in rows], dtype=float)
    finite = scores[np.isfinite(scores)]
    return {'pairs': len(rows), 'scenes': len({row['image_id'] for row in rows}),
        'psnr_db': {'mean': float(finite.mean()), 'median': float(np.median(finite)),
                    'minimum': float(finite.min()), 'q05': float(np.quantile(finite, .05)),
                    'q10': float(np.quantile(finite, .10)), 'maximum': float(finite.max())},
        'mean_pixel_mae_255': float(np.mean([row['pixel_mae_255'] for row in rows])),
        'control_statistics': control_statistics(rows)}


def render_histograms(rows, retained_rows, cutoffs, output):
    values = np.asarray([row['psnr_db'] for row in rows if np.isfinite(row['psnr_db'])])
    retained_values = np.asarray([row['psnr_db'] for row in retained_rows if np.isfinite(row['psnr_db'])])
    bins = np.histogram_bin_edges(values, bins=32)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].hist(values, bins=bins, color='#277f89', alpha=.88, edgecolor='white')
    axes[0].set(title=f'All {len(values)} expert-fit pairs', xlabel='PSNR, fit vs expert display (dB)',
                ylabel='Pairs')
    for fraction, cutoff in cutoffs.items():
        axes[0].axvline(cutoff, linestyle='--', linewidth=1.6,
                        label=f'{fraction:.0%} trim cutoff: {cutoff:.2f} dB')
    axes[0].legend()
    axes[1].hist(retained_values, bins=bins, color='#d17a4a', alpha=.88, edgecolor='white')
    axes[1].set(title='Retained after lowest-5% trim', xlabel='PSNR, fit vs expert display (dB)',
                ylabel='Pairs')
    axes[1].axvline(cutoffs[.05], color='#8c3d2e', linestyle='--', linewidth=1.6,
                    label=f'Removal boundary: {cutoffs[.05]:.2f} dB')
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)

def render_control_histograms(all_path, retained_path, output):
    all_rows = list(csv.DictReader(all_path.open(newline='')))
    retained_rows = list(csv.DictReader(retained_path.open(newline='')))
    if not all_rows or not retained_rows:
        raise ValueError('Missing control rows for histogram comparison')
    full_bins = np.linspace(-1, 1, 41)
    colors = dict(zip('ABCDE', ('#236a83', '#dc7650', '#55976b', '#8169a9', '#c19b32')))
    titles = {'warmth': 'Warmth', 'saturation': 'Saturation', 'tone_gain': 'Brightness / tone gain',
              'contrast': 'Contrast', 'sharpness': 'Sharpness', 'palette': 'Palette'}
    figure, axes = plt.subplots(2, 6, figsize=(21, 8), sharex=True, constrained_layout=True)
    for column, control in enumerate(CONTROL_NAMES):
        key = f'{control}_p90'
        for row_index, (rows, label) in enumerate(((all_rows, 'All 625 fits'),
                                                   (retained_rows, f'After PSNR trim (n={len(retained_rows)})'))):
            axis = axes[row_index, column]
            for expert in 'ABCDE':
                values = np.asarray([float(item[key]) for item in rows if item['expert'] == expert])
                axis.hist(values, bins=full_bins, histtype='step', linewidth=1.35,
                          color=colors[expert], label=expert)
            axis.axvline(0, color='#33444f', linewidth=.8, alpha=.7)
            axis.set_title(f'{titles[control]} | {label}')
            axis.set_xlim(-1, 1)
            axis.set_xlabel('P90-normalized fit value')
            if column == 0:
                axis.set_ylabel('Expert-fit pairs')
            if row_index == 0 and column == 2:
                axis.legend(title='Expert', ncol=5, fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return len(all_rows), len(retained_rows)

def refresh_control_histograms(output):
    all_count, retained_count = render_control_histograms(
        output / 'all_pairs.csv', output / 'retained_after_5pct_trim.csv',
        output / 'control_histograms_all_vs_psnr_trim.png')
    index = output / 'index.html'
    text = index.read_text()
    if 'control_histograms_all_vs_psnr_trim.png' not in text:
        text = text.replace('</ul>', '</ul><h2>ISP-control distributions by expert</h2>'
            '<p>Top: all fit pairs. Bottom: retained after removing lowest-PSNR 5%. '
            'Both use identical P90-normalized bins over [-1,1].</p>'
            '<img style="max-width:100%" src="control_histograms_all_vs_psnr_trim.png">')
        index.write_text(text)
    print(f'Created per-control histograms: {all_count} full pairs, {retained_count} retained')

def run(args):
    latest = json.loads((args.dataset / 'latest.json').read_text())
    release = args.dataset / latest['path']
    release_manifest = json.loads((release / 'dataset_manifest.json').read_text())
    rows = load_psnr_pairs(release)
    if len(rows) != release_manifest['expert_fit_pairs']:
        raise ValueError('Scored pair count differs from the release manifest')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    finite_scores = sorted(row['psnr_db'] for row in rows if np.isfinite(row['psnr_db']))
    cutoffs = {fraction: finite_scores[math.ceil(len(finite_scores) * fraction) - 1]
               for fraction in TRIM_FRACTIONS}
    retained_5, excluded_5 = trim_lowest(rows, .05)
    retained_10, excluded_10 = trim_lowest(rows, .10)
    write_json(args.out / 'control_summary_all.json', summarise(rows))
    write_json(args.out / 'control_summary_trimmed_5pct.json', summarise(retained_5))
    write_json(args.out / 'control_summary_trimmed_10pct.json', summarise(retained_10))
    write_json(args.out / 'excluded_lowest_5pct.json', [
        {key: row[key] for key in ('image_id', 'expert', 'psnr_db', 'pixel_mae_255',
                                    'normalized_controls', 'physical_controls')} for row in excluded_5])
    write_json(args.out / 'excluded_lowest_10pct.json', [
        {key: row[key] for key in ('image_id', 'expert', 'psnr_db', 'pixel_mae_255',
                                    'normalized_controls', 'physical_controls')} for row in excluded_10])
    for name, subset in (('all_pairs.csv', rows), ('retained_after_5pct_trim.csv', retained_5)):
        with (args.out / name).open('w', newline='') as stream:
            flattened = flatten_control_rows(subset)
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    render_histograms(rows, retained_5, cutoffs, args.out / 'psnr_histograms.png')
    summary = {'release': latest['release'], 'release_path': str(release), 'pairs': len(rows),
        'metric': 'PSNR computed on saved 8-bit sRGB expert-display vs fitted-ISP render pairs',
        'trim_rule': 'Rank individual expert-fit pairs by PSNR ascending; remove ceil(fraction * pair count).',
        'trim_cutoffs_db': {f'{fraction:.0%}': cutoff for fraction, cutoff in cutoffs.items()},
        'all_pairs': summarise(rows), 'retained_after_5pct': summarise(retained_5),
        'retained_after_10pct': summarise(retained_10),
        'excluded_5pct': len(excluded_5), 'excluded_10pct': len(excluded_10),
        'interpretation': 'Trimmed statistics are conditional fit-setting descriptions, not an unbiased full-dataset summary. PSNR trimming selects easier expert-fit pairs and does not validate perceptual quality or sharpness.'}
    write_json(args.out / 'summary.json', summary)
    (args.out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>FiveK ISP PSNR-trimmed statistics</title>'
        '<h1>FiveK ISP tuning statistics, PSNR trimmed</h1>'
        '<p>PSNR is on saved 8-bit display renders, not full-resolution images. The 5% trim removes 32 of 625 pairs.</p>'
        '<img style="max-width:100%" src="psnr_histograms.png">'
        '<ul><li><a href="summary.json">All, 5%-trimmed, and 10%-trimmed summaries</a></li>'
        '<li><a href="control_summary_trimmed_5pct.json">Per-control means by expert after 5% trim</a></li>'
        '<li><a href="all_pairs.csv">All fits and tuning values</a></li>'
        '<li><a href="retained_after_5pct_trim.csv">Fits retained after 5% trim</a></li>'
        '<li><a href="excluded_lowest_5pct.json">Excluded lowest-PSNR pairs</a></li>'
        '<li><a href="excluded_lowest_10pct.json">Excluded lowest-10%-PSNR sensitivity set</a></li></ul>'
        f'<p>5% boundary: {cutoffs[.05]:.3f} dB; retained: {len(retained_5)} pairs.</p>')
    print(json.dumps({'output': str(args.out), 'pairs': len(rows), 'trimmed_5pct': len(excluded_5),
                      'cutoff_5pct_db': cutoffs[.05], 'retained': len(retained_5)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--refresh-control-histograms', action='store_true')
    args = parser.parse_args()
    if args.refresh_control_histograms:
        refresh_control_histograms(args.out)
    else:
        run(args)

if __name__ == '__main__':
    main()
"""Rank saved FiveK expert-fit renders by display-RGB PSNR and make examples."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SOURCE = Path('/data/data1/FiveK/C_data/results/isp_match_105_v4')
DEFAULT_OUT = Path('/data/data1/FiveK/C_data/results/isp_match_105_fit_psnr_audit_v1')
CONTROLS = ('warmth', 'saturation', 'tone_gain', 'contrast', 'sharpness', 'palette')
PANEL_W, PANEL_H, ROW_H, MARGIN = 260, 185, 295, 20


def font(size):
    for candidate in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def evaluate_pair(fit, image_id, expert):
    with Image.open(fit['display_path']) as image:
        target = np.asarray(image.convert('RGB'), dtype=np.float64)
    with Image.open(fit['fit_display_path']) as image:
        predicted = np.asarray(image.convert('RGB'), dtype=np.float64)
    if target.shape != predicted.shape:
        raise ValueError(f'Image sizes differ for {image_id}/{expert}: {target.shape} vs {predicted.shape}')
    mse = float(np.mean((target - predicted) ** 2))
    return {'image_id': image_id, 'expert': expert, 'psnr_db': float('inf') if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
            'mse_8bit_squared': mse, 'mae_8bit': float(np.mean(np.abs(target - predicted))),
            'fit_loss': float(fit['loss']), 'best_iteration': int(fit['iteration']),
            'dimensions_wh': [int(target.shape[1]), int(target.shape[0])],
            'physical_controls': fit['physical_controls'], 'normalized_controls': fit['normalized_controls'],
            'expert_display_path': fit['display_path'], 'default_display_path': fit['neutral_display_path'],
            'fit_display_path': fit['fit_display_path']}


def thumbnail(canvas, path, position, size):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail(size, Image.Resampling.LANCZOS)
        x, y = position
        canvas.paste(image, (x + (size[0] - image.width) // 2, y + (size[1] - image.height) // 2))


def render_sheet(rows, output, title):
    width = MARGIN * 2 + 4 * PANEL_W
    height = MARGIN * 2 + 65 + len(rows) * ROW_H
    canvas = Image.new('RGB', (width, height), '#f5f7f7')
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 14), title, fill='#172a3a', font=font(23))
    draw.text((MARGIN, 45), 'Target: Expert TIFF preview | Fit is compared at saved 720p render resolution',
              fill='#526878', font=font(13))
    labels = ('Expert TIFF target', 'Default ISP', 'Fitted ISP')
    for row_index, row in enumerate(rows):
        y = MARGIN + 65 + row_index * ROW_H
        draw.rectangle((MARGIN, y, width - MARGIN, y + ROW_H - 8), fill='white', outline='#d5e0e5')
        draw.text((MARGIN + 8, y + 6),
                  f"{row['image_id']} | Expert {row['expert']} | PSNR {row['psnr_db']:.2f} dB | "
                  f"MAE {row['mae_8bit']:.2f}/255 | fit loss {row['fit_loss']:.5f}",
                  fill='#172a3a', font=font(13))
        for column, (label, key) in enumerate(zip(labels, ('expert_display_path', 'default_display_path', 'fit_display_path'))):
            x = MARGIN + column * PANEL_W
            draw.text((x + 6, y + 27), label, fill='#526878', font=font(12))
            thumbnail(canvas, row[key], (x + 5, y + 48), (PANEL_W - 12, 180))
        x = MARGIN + 3 * PANEL_W + 8
        p90 = row['normalized_controls']
        lines = [f"{name}: {p90[name]:+.3f}" for name in CONTROLS]
        lines += [f"Best iteration: {row['best_iteration']}",
                  f"Render size: {row['dimensions_wh'][0]}x{row['dimensions_wh'][1]}"]
        draw.multiline_text((x, y + 32), '\n'.join(lines), fill='#263b47', font=font(12), spacing=5)
    canvas.save(output, optimize=True)


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    index = json.loads((args.source / 'index.json').read_text())
    rows = []
    for entry in index['images']:
        result = json.loads(Path(entry['results']).read_text())
        for expert, fit in result['per_image'][entry['image_id']].items():
            rows.append(evaluate_pair(fit, entry['image_id'], expert))
    if len(rows) != index['completed_pairs']:
        raise ValueError('PSNR audit pair count differs from source index')
    ordered = sorted(rows, key=lambda row: row['psnr_db'])
    worst, best = ordered[:args.examples], ordered[-args.examples:][::-1]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, examples in (('worst_psnr', worst), ('best_psnr', best)):
        render_sheet(examples, args.out / f'{name}.png',
                     f'{name.replace("_", " ").title()} fitted renders vs expert targets')
    fields = ['image_id', 'expert', 'psnr_db', 'mse_8bit_squared', 'mae_8bit', 'fit_loss',
              'best_iteration', 'dimensions_wh', 'normalized_controls', 'physical_controls',
              'expert_display_path', 'default_display_path', 'fit_display_path']
    with (args.out / 'all_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / 'worst_best.json').open('w') as stream:
        json.dump({'metric': 'PSNR on 8-bit saved display RGB, fit vs Expert TIFF preview',
                   'mean_psnr_db': float(np.mean([row['psnr_db'] for row in rows])),
                   'median_psnr_db': float(np.median([row['psnr_db'] for row in rows])),
                   'pairs': len(rows), 'worst': worst, 'best': best}, stream, indent=2, allow_nan=False)
    (args.out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>FiveK Fit PSNR Audit</title>'
        '<h1>FiveK fit PSNR audit</h1><p>PSNR is measured on saved 8-bit sRGB renders at 720p resolution; '
        'it is not a full-resolution sharpness or perceptual-quality measure.</p>'
        '<p><a href="all_pairs.csv">All 525 scored fits and controls</a> | '
        '<a href="worst_best.json">Worst/best records</a></p>'
        f'<h2>Worst {len(worst)}</h2><img style="max-width:100%" src="worst_psnr.png">'
        f'<h2>Best {len(best)}</h2><img style="max-width:100%" src="best_psnr.png">')
    print(json.dumps({'output': str(args.out), 'pairs': len(rows), 'mean_psnr_db': float(np.mean([row['psnr_db'] for row in rows])),
                      'worst': [(row['image_id'], row['expert'], round(row['psnr_db'], 2)) for row in worst],
                      'best': [(row['image_id'], row['expert'], round(row['psnr_db'], 2)) for row in best]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--examples', type=int, default=5)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error('--examples must be positive')
    run(args)


if __name__ == '__main__':
    main()
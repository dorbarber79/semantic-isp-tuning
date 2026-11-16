"""Fit one FiveK scene and export native-resolution renders for visual review."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import rawpy
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
SEMANTIC_ISP_V2 = WORKSPACE / 'experiments/semantic_isp/v2'
if str(SEMANTIC_ISP_V2) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ISP_V2))

from axis_isp import linear_to_srgb8
from differentiable_isp import load_and_preprocess_raw
from moran_semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints
from fivek_photographer_fit import fit_controls, image_statistics, prophoto_tiff_to_linear_srgb


DEFAULT_RELEASE = Path('/data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-15-v5')
DEFAULT_CONFIG = SEMANTIC_ISP_V2 / 'calibration/exp81/direct_soft_calibration.json'
DEFAULT_OUTPUT = Path('/data/data1/FiveK/C_data/results/fullres_annotation_a3491_v1')
EXIFTOOL_TAGS = ('Make', 'Model', 'ExposureTime', 'ISO', 'FNumber', 'ExposureCompensation',
                 'BrightnessValue', 'WhiteBalance', 'ColorTemperature', 'Tint', 'Exposure',
                 'Brightness', 'Contrast', 'ToneCurveName', 'CameraProfile')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exif_metadata(path, exiftool):
    result = subprocess.run([exiftool, '-json', *[f'-{tag}' for tag in EXIFTOOL_TAGS], str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)[0]


def rawpy_metadata(path):
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        return {
            'raw_dimensions_wh': [int(sizes.width), int(sizes.height)],
            'visible_dimensions_wh': [int(sizes.iwidth), int(sizes.iheight)],
            'white_level': int(raw.white_level),
            'black_level_per_channel': [int(value) for value in raw.black_level_per_channel],
            'camera_whitebalance': [float(value) for value in raw.camera_whitebalance],
            'raw_pattern': raw.raw_pattern.tolist(),
            'color_description': raw.color_desc.decode('ascii', errors='replace'),
        }


def raw_p99_audit(path):
    with rawpy.imread(str(path)) as raw:
        decoded = raw.postprocess(
            use_camera_wb=True, gamma=(1, 1), no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB, output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, use_auto_wb=False,
            half_size=False, highlight_mode=rawpy.HighlightMode.Clip)
    decoded_float = decoded.astype(np.float32) / 65535.0
    p99 = float(np.percentile(decoded_float, 99))
    gain = float(np.clip(.9 / p99, 1.0, 10.0)) if p99 > .01 else 1.0
    return {'postprocess_rgb_p99_before_gain': p99,
            'applied_p99_gain': gain,
            'gain_stops': float(np.log2(gain)),
            'clipped_rgb_sample_fraction_after_gain': float(np.mean(decoded_float * gain >= 1.0))}


def write_linear16_png(path, image):
    rgb = image[0].detach().cpu().permute(1, 2, 0).numpy()
    encoded = np.rint(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not write linear 16-bit PNG: {path}')


def write_display8_png(path, image):
    rgb = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not write display 8-bit PNG: {path}')


def file_record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def load_v5_scene(release, image_id):
    pairs = json.loads((release / 'annotations/normalized_pairs.json').read_text())
    rows = [row for row in pairs if row['image_id'] == image_id]
    if {row['expert'] for row in rows} != set('ABCDE'):
        raise ValueError(f'Expected the five A-E expert assets in release scene {image_id}')
    first = rows[0]
    raw = release / first['assets']['raw_original']['path']
    targets = {row['expert']: release / row['assets']['expert_original']['path'] for row in rows}
    return raw, targets


def run(args):
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {args.out}')
    release = args.release.resolve()
    if not (release / 'dataset_manifest.json').is_file():
        raise FileNotFoundError(f'Not a FiveK-ISPVA release: {release}')
    manifest = json.loads((release / 'dataset_manifest.json').read_text())
    if manifest.get('publication_status', '').startswith('Public'):
        raise ValueError('This experiment expects the internal MIT-Adobe FiveK release with original masters')
    raw_path, target_paths = load_v5_scene(release, args.image_id)
    config = json.loads(args.config.read_text())
    endpoints = pole_endpoints(config)
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.set_num_threads(2)

    raw_info = rawpy_metadata(raw_path)
    capture_metadata = exif_metadata(raw_path, args.exiftool)
    p99_info = raw_p99_audit(raw_path)
    neutral = load_and_preprocess_raw(str(raw_path), None).to(device=device, dtype=torch.float32)
    dimensions = [int(neutral.shape[-1]), int(neutral.shape[-2])]
    output = args.out
    output.mkdir(parents=True)
    write_linear16_png(output / 'default_linear_16bit.png', neutral)
    write_display8_png(output / 'default_display_8bit.png', neutral)

    baseline = {'experiment': 'FiveK native-resolution fit-render annotation pilot',
        'version': 'fullres_annotation_a3491_v1', 'image_id': args.image_id,
        'source_release': str(release), 'source_release_name': manifest['release'],
        'default_definition': 'Native-size rawpy V1 preprocessing of the packaged DNG; this is NOT the original precomputed FiveK Default ISP rendering.',
        'target_definition': 'The five original 16-bit ProPhoto RGB expert TIFF masters A-E, ICC converted to linear sRGB.',
        'raw_source': file_record(raw_path), 'expert_sources': {expert: file_record(path)
            for expert, path in target_paths.items()},
        'capture_metadata_exiftool': capture_metadata,
        'rawpy_sensor_metadata': raw_info, 'v1_raw_preprocessing_p99_audit': p99_info,
        'raw_preprocess': {'camera_white_balance': True, 'demosaic': 'AHD', 'gamma': [1, 1],
            'no_auto_bright': True, 'output_color': 'sRGB primaries, linear transfer',
            'rawpy_output_bits_per_channel': 16, 'highlight_mode': 'Clip',
            'p99_normalization': 'scale combined RGB sample p99 to 0.9, clamp gain to [1,10], clamp RGB to [0,1]',
            'denoise': 'OpenCV bilateralFilter diameter 5, sigmaColor 0.05, sigmaSpace 35'},
        'native_dimensions_wh': dimensions,
        'optimizer': {'iterations': args.iterations, 'learning_rate': args.learning_rate,
            'optimization_short_side': 256, 'palette_starts': [0.0, -0.12, 0.12],
            'device': str(device), 'renderer_config_sha256': sha256(args.config)},
        'exports': {'default_linear_16bit_png': 'lossless PNG compression; values quantized to uint16 linear RGB',
            'default_display_8bit_png': 'lossless PNG compression; sRGB transfer and uint8 quantization',
            'fit_linear_16bit_png': 'lossless PNG compression; values quantized to uint16 linear RGB',
            'fit_display_8bit_png': 'lossless PNG compression; sRGB transfer and uint8 quantization',
            'expert_tiffs': 'original TIFF masters are referenced in-place by path and SHA-256, not copied'},
        'annotation_protocol': {'description_target': 'fit versus generated native-size RAW baseline; expert TIFF is contextual reference',
            'render_dimensions_wh': dimensions, 'full_resolution_visual_review_required': True,
            'pixel_crops_reviewed': False, 'model_identifier': 'Not yet recorded'},
        'files': {}}

    fit_records = {}
    for expert, tiff_path in target_paths.items():
        target_array, profile_info = prophoto_tiff_to_linear_srgb(tiff_path, verify_icc=True)
        target = torch.from_numpy(np.ascontiguousarray(target_array)).permute(2, 0, 1).unsqueeze(0).to(device)
        target = target.clamp(0, 1)
        if target.shape[-2:] != neutral.shape[-2:]:
            target = torch.nn.functional.interpolate(target, size=neutral.shape[-2:], mode='area').clamp(0, 1)
        fit = fit_controls(neutral, target, config, endpoints, args.iterations, args.learning_rate)
        physical = torch.tensor([fit['physical_controls'][name] for name in CONTROL_NAMES],
                                device=device, dtype=neutral.dtype)
        with torch.no_grad():
            native_fit = render_linear(neutral, parameters_from_controls(physical, config))
        write_linear16_png(output / f'expert_{expert}_target_linear_16bit.png', target)
        write_display8_png(output / f'expert_{expert}_target_display_8bit.png', target)
        write_linear16_png(output / f'expert_{expert}_fit_linear_16bit.png', native_fit)
        write_display8_png(output / f'expert_{expert}_fit_display_8bit.png', native_fit)
        fit_records[expert] = {'expert_source': file_record(tiff_path), 'icc_transform': profile_info,
            'controls': fit, 'target_statistics': image_statistics(target),
            'fit_statistics': image_statistics(native_fit),
            'exports': {suffix: file_record(output / f'expert_{expert}_{suffix}.png')
                for suffix in ('target_linear_16bit', 'target_display_8bit',
                               'fit_linear_16bit', 'fit_display_8bit')}}
        del target, native_fit
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    baseline['fits'] = fit_records
    baseline['files'] = {name: file_record(output / name)
        for name in ('default_linear_16bit.png', 'default_display_8bit.png')}
    (output / 'experiment.json').write_text(json.dumps(baseline, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'image_id': args.image_id,
        'native_dimensions_wh': dimensions, 'expert_fits': len(fit_records),
        'annotation_ready': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--image-id', default='a3491-DSC_0091')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=.03)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exiftool', default='exiftool')
    args = parser.parse_args()
    if args.iterations < 1 or args.learning_rate <= 0:
        parser.error('Iterations and learning rate must be positive')
    run(args)


if __name__ == '__main__':
    main()
er Moran's five bipolar ISP axes on the ten selected RAW image

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from axis_isp import MoranAxisISP, linear_to_srgb8
from differentiable_isp import load_and_preprocess_raw

DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
OUT = Path("/data/data1/semantic_isp/results/exp31_moran_axis_sweep_srgb8_tuned")
IMAGES = {
    1: "desert", 28: "dog", 2: "child", 236: "man", 194: "swan",
    277: "snow", 263: "dusk", 384: "backlit", 33: "indoor", 103: "city",
}
CELL_W, CELL_H, ROW_LABEL, TOP = 192, 128, 72, 58


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def coordinate_name(value):
    return f"{'p' if value >= 0 else 'm'}{abs(value):.3f}"


def to_preview(image_srgb8):
    pixels = image_srgb8[0].detach().cpu().permute(1, 2, 0).numpy()
    resized = cv2.resize(pixels, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def save_srgb8(path, image_srgb8):
    pixels = image_srgb8[0].detach().cpu().permute(1, 2, 0).numpy()
    if pixels.dtype != np.uint8:
        raise TypeError("Expected final 8-bit sRGB output")
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise OSError(f"Failed to write 8-bit sRGB PNG: {path}")


def make_contact_sheet(axis, previews, output_path, images):
    coordinates = MoranAxisISP.AXIS_STEPS
    width = ROW_LABEL + len(coordinates) * CELL_W
    height = TOP + len(images) * CELL_H
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    small, label = font(13), font(14)
    negative_pole, positive_pole = MoranAxisISP.POLES[axis]
    draw.text((ROW_LABEL, 5), f"{axis}: {negative_pole}  <- coordinate ->  {positive_pole}",
              fill="black", font=label)
    for col, value in enumerate(coordinates):
        x = ROW_LABEL + col * CELL_W
        draw.text((x + 5, 31), f"{value:+.2f}", fill="black", font=small)
    for row, image_id in enumerate(images):
        scene = IMAGES[image_id]
        y = TOP + row * CELL_H
        draw.text((4, y + 8), f"{image_id}: {scene}", fill="black", font=small)
        for col, value in enumerate(coordinates):
            sheet.paste(previews[(image_id, value)], (ROW_LABEL + col * CELL_W, y))
    sheet.save(output_path, quality=92)


def run(images, output, device):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    rows = []
    previews = {axis: {} for axis in MoranAxisISP.AXES}
    for image_id in images:
        dng = DATA / f"{image_id}.dng"
        if not dng.is_file():
            raise FileNotFoundError(dng)
        image = load_and_preprocess_raw(str(dng), None).to(device)
        image_dir = output / f"img_{image_id}"
        image_dir.mkdir(parents=True, exist_ok=True)
        isp = MoranAxisISP(device)
        with torch.no_grad():
            neutral = isp(image)
            neutral_srgb8 = linear_to_srgb8(neutral)
            save_srgb8(image_dir / "neutral.png", neutral_srgb8)
            neutral_luma = (0.299 * neutral[:, 0] + 0.587 * neutral[:, 1]
                            + 0.114 * neutral[:, 2]).mean().item()
            for axis in MoranAxisISP.AXES:
                axis_dir = image_dir / axis
                axis_dir.mkdir(exist_ok=True)
                for coordinate in MoranAxisISP.AXIS_STEPS:
                    isp.set_axes({axis: coordinate})
                    rendered = isp(image)
                    rendered_srgb8 = linear_to_srgb8(rendered)
                    name = coordinate_name(coordinate)
                    save_srgb8(axis_dir / f"{name}.png", rendered_srgb8)
                    previews[axis][(image_id, coordinate)] = to_preview(rendered_srgb8)
                    mean_luma = (0.299 * rendered[:, 0].mean()
                                 + 0.587 * rendered[:, 1].mean()
                                 + 0.114 * rendered[:, 2].mean()).item()
                    rows.append({
                        "image": image_id, "scene": IMAGES[image_id], "axis": axis,
                        "coordinate": coordinate,
                        "pole": ("Neutral" if coordinate == 0 else
                                 MoranAxisISP.POLES[axis][1 if coordinate > 0 else 0]),
                        "mean_linear_luma": mean_luma,
                        "neutral_mean_linear_luma": neutral_luma,
                        "delta_mean_linear_luma": mean_luma - neutral_luma,
                        "min_rgb": rendered.min().item(), "max_rgb": rendered.max().item(),
                        "file": str((axis_dir / f"{name}.png").relative_to(output)),
                    })
                    del rendered
        del image, neutral, isp
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"finished image {image_id}: {IMAGES[image_id]}", flush=True)

    for axis in MoranAxisISP.AXES:
        make_contact_sheet(axis, previews[axis], output / f"axis_{axis}.jpg", images)
    with (output / "sweep_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "description": "One axis varied at a time from -0.4 to +0.4; all other axes set to zero.",
        "images": {str(key): value for key, value in IMAGES.items() if key in images},
        "axis_steps": list(MoranAxisISP.AXIS_STEPS),
        "poles": MoranAxisISP.POLES,
        "control_deltas_per_coordinate": MoranAxisISP.AXIS_DELTAS,
        "moral_darkness_endpoint": MoranAxisISP.MORAL_DARKNESS_ENDPOINT,
        "vitality_monochrome_endpoint": MoranAxisISP.VITALITY_MONOCHROME_ENDPOINT,
        "vitality_saturation_delta": MoranAxisISP.VITALITY_SATURATION_DELTA,
        "palette_axis_swatch_pairs": MoranAxisISP.PALETTE_SWATCHES,
        "palette_axis_points_yiq": MoranAxisISP("cpu").palette_axis_points.tolist(),
        "palette_axis_max_perpendicular_attraction": MoranAxisISP.HUE_AXIS_MAX_ATTRACTION,
        "palette_axis_max_shift_fraction": MoranAxisISP.HUE_AXIS_SHIFT_FRACTION,
        "domain": "linear RGB from RAW decode through sharpening; sRGB output transfer and 8-bit quantization occur once after the final ISP stage",
        "processing_dtype": "float32; no intermediate integer quantization",
        "export": "native-resolution, lossless 8-bit sRGB RGB PNG",
        "device": str(device),
        "source_defaults": {key: float(value) for key, value in
                            zip(("wb_warmth", "saturation", "brightness", "sharpness", "contrast"),
                                MoranAxisISP("cpu").defaults.tolist())},
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"images": len(images), "renders_per_axis": len(rows) // len(MoranAxisISP.AXES),
                      "total_sweeps": len(rows), "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=int, default=list(IMAGES))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    unknown = set(args.images) - set(IMAGES)
    if unknown:
        parser.error(f"Unknown image IDs: {sorted(unknown)}")
    run(args.images, args.out, args.device)


if __name__ == "__main__":
    main()

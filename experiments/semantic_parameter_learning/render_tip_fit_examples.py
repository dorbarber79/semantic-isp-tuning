"""Render a TIP figure showing Default ISP and fitted FiveK outputs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data/data1/FiveK/C_data/results/isp_match_105_v4/images")
OUTPUT = Path(__file__).resolve().parents[2] / "paper_figures" / "fivek_fit_examples.png"
SCENES = (
    ("a0015-DSC_0081", "Beach portrait"),
    ("a0017-050710_031618__MG_3496", "Woodland shelter"),
)
PANEL_LABELS = ("Default ISP", "Fit A", "Fit B", "Fit C", "Fit D", "Fit E")
PANEL_WIDTH = 300
PANEL_HEIGHT = 225
ROW_HEIGHT = 330
MARGIN = 24


def load_font(size):
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width = MARGIN * 2 + PANEL_WIDTH * len(PANEL_LABELS)
    height = MARGIN + ROW_HEIGHT * len(SCENES) + MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    label_font = load_font(16)

    for row_index, (image_id, scene_label) in enumerate(SCENES):
        scene_root = ROOT / image_id
        top = MARGIN + row_index * ROW_HEIGHT
        draw.text((MARGIN, top), f"{image_id}: {scene_label}", fill="#1f2c34", font=title_font)
        image_top = top + 55
        paths = [scene_root / "targets" / f"{image_id}_neutral.png"]
        paths.extend(scene_root / "renders" / f"{image_id}_expert_{letter}_isp_fit.png"
                     for letter in "ABCDE")
        for column, (label, path) in enumerate(zip(PANEL_LABELS, paths)):
            left = MARGIN + column * PANEL_WIDTH
            draw.text((left + 4, image_top), label, fill="#34444c", font=label_font)
            with Image.open(path) as source:
                photo = source.convert("RGB")
                photo.thumbnail((PANEL_WIDTH - 12, PANEL_HEIGHT), Image.Resampling.LANCZOS)
                x = left + (PANEL_WIDTH - photo.width) // 2
                y = image_top + 28 + (PANEL_HEIGHT - photo.height) // 2
                canvas.paste(photo, (x, y))

    canvas.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT} ({width}x{height})")


if __name__ == "__main__":
    render()
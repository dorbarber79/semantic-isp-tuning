"""Probe image-based FiveK edit descriptions with a local vision-language model."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
SOURCE = Path("/data/data1/FiveK/C_data/results/isp_match_105_v3")


def pair_image(default_path, expert_path):
    with Image.open(default_path) as source, Image.open(expert_path) as target:
        left, right = source.convert("RGB"), target.convert("RGB")
    if left.size != right.size:
        raise ValueError("Default and expert previews must have equal dimensions")
    left.thumbnail((600, 650), Image.Resampling.LANCZOS)
    right.thumbnail((600, 650), Image.Resampling.LANCZOS)
    width = max(left.width, right.width) + 24
    height = max(left.height, right.height) + 60
    sheet = Image.new("RGB", (2 * width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    draw.text((10, 10), "BEFORE: DEFAULT", fill="black", font=font)
    draw.text((width + 10, 10), "AFTER: EXPERT", fill="black", font=font)
    sheet.paste(left, ((width - left.width) // 2, 50))
    sheet.paste(right, (width + (width - right.width) // 2, 50))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default="a0015-DSC_0081")
    parser.add_argument("--expert", choices="ABCDE", default="E")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--representation", choices=("separate", "sheet"), default="separate")
    args = parser.parse_args()
    path = SOURCE / "images" / args.image_id / "results.json"
    fit = json.loads(path.read_text())["per_image"][args.image_id][args.expert]
    if args.representation == "sheet":
        images = [pair_image(fit["neutral_display_path"], fit["display_path"])]
        image_content = [{"type": "image"}]
        image_order = "BEFORE on the left with AFTER on the right"
    else:
        with Image.open(fit["neutral_display_path"]) as original, Image.open(fit["display_path"]) as edited:
            images = [original.convert("RGB"), edited.convert("RGB")]
        if images[0].size != images[1].size:
            raise ValueError("Default and expert previews must have equal dimensions")
        image_content = [{"type": "image"}, {"type": "image"}]
        image_order = "the first image (BEFORE) with the second image (AFTER)"
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                              min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, attn_implementation="sdpa")
    model.eval()
    request = (f"Compare {image_order}. They show the same photo. "
               "Describe only visible changes in lighting, color, and overall appearance in AFTER relative to BEFORE. "
               "Mention uncertainty if differences are subtle. Do not invent objects, state camera settings, "
               "or guess the photographer's intent. Answer in at most three short sentences.")
    messages = [{"role": "user", "content": [*image_content, {"type": "text", "text": request}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(json.dumps({"image_id": args.image_id, "expert": args.expert, "model": MODEL,
                      "revision": REVISION, "representation": args.representation,
                      "prompt": request, "visual_observation": answer}, indent=2))


if __name__ == "__main__":
    main()
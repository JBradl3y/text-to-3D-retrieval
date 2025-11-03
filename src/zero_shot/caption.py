"""Generate BLIP captions for rendered views and keep high CLIPScore results."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

from .config import ZeroShotPaths


def parse_args():
    parser = argparse.ArgumentParser(description="Caption rendered views and score with CLIP.")
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--checkpoint-blip", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--checkpoint-clip", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--min-clip-score", type=float, default=0.25)
    parser.add_argument("--max-captions", type=int, default=3)
    return parser.parse_args()


def clean_caption(text):
    return text.strip().lower()


def load_models(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    blip_processor = BlipProcessor.from_pretrained(args.checkpoint_blip)
    blip_model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_blip).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.checkpoint_clip)
    clip_model = CLIPModel.from_pretrained(args.checkpoint_clip).to(device)
    blip_model.eval()
    clip_model.eval()
    return device, blip_processor, blip_model, clip_processor, clip_model


def generate_caption(image, processor, model, device, max_length):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return clean_caption(processor.decode(output[0], skip_special_tokens=True))


def clip_score(image, caption, processor, model, device):
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_emb * text_emb).sum(dim=-1).cpu().item())


def caption_views(render_folder, blip_processor, blip_model, clip_processor, clip_model, device, max_length):
    outputs = []
    for image_path in sorted(render_folder.glob("view_*.png")):
        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image, blip_processor, blip_model, device, max_length)
        if not caption:
            continue
        score = clip_score(image, caption, clip_processor, clip_model, device)
        outputs.append({"image_path": str(image_path), "caption": caption, "clip_score": score})
    return outputs


def main():
    args = parse_args()
    paths = ZeroShotPaths.from_env()
    render_dir = Path(args.render_dir) if args.render_dir else paths.render_dir
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else paths.captions_jsonl
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device, blip_processor, blip_model, clip_processor, clip_model = load_models(args)

    df = pd.read_csv(args.index_csv)
    df["category"] = df["category"].str.lower()

    with output_jsonl.open("w") as writer:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Captioning shapes"):
            model_id = str(row["modelId"])
            category = row["category"]
            render_folder = Path(render_dir) / category / model_id

            if not render_folder.exists():
                writer.write(json.dumps({"modelId": model_id, "category": category, "status": "missing_renders", "captions": []}) + "\n")
                continue

            captions = caption_views(render_folder, blip_processor, blip_model, clip_processor, clip_model, device, args.max_length)
            captions = [c for c in captions if c["clip_score"] >= args.min_clip_score]
            captions.sort(key=lambda x: x["clip_score"], reverse=True)
            if args.max_captions > 0:
                captions = captions[: args.max_captions]

            status = "ok" if captions else "no_caption"
            writer.write(json.dumps({"modelId": model_id, "category": category, "status": status, "captions": captions}) + "\n")

    print(f"Wrote caption records to {output_jsonl}")


if __name__ == "__main__":
    main()

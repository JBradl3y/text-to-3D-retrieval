"""Convert caption JSONL into a CLIP-embedded retrieval index."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer

from .config import ZeroShotPaths


def parse_args():
    parser = argparse.ArgumentParser(description="Build zero-shot retrieval index.")
    parser.add_argument("--captions-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--checkpoint-clip", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def encode_caption(text, model, tokenizer, device):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.squeeze(0).cpu()


def main():
    args = parse_args()
    paths = ZeroShotPaths.from_env()
    captions_path = Path(args.captions_jsonl) if args.captions_jsonl else paths.captions_jsonl
    output_csv = Path(args.output_csv) if args.output_csv else paths.retrieval_index
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    clip_model = CLIPModel.from_pretrained(args.checkpoint_clip).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint_clip)
    clip_model.eval()

    records = []
    with captions_path.open() as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("status") != "ok" or not entry.get("captions"):
                continue
            best = entry["captions"][0]
            embedding = encode_caption(best["caption"], clip_model, clip_tokenizer, device)
            records.append(
                {
                    "modelId": entry["modelId"],
                    "category": entry["category"],
                    "caption": best["caption"],
                    "clip_embedding": ",".join(map(str, embedding.tolist())),
                    "clip_score": best["clip_score"],
                    "image_path": best["image_path"],
                }
            )

    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"Wrote {len(records)} entries to {output_csv}")


if __name__ == "__main__":
    main()

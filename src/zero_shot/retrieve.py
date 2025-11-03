"""Rank Text2Shape models for a natural-language query using stored CLIP embeddings."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer

from .config import ZeroShotPaths


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot text-to-shape retrieval.")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint-clip", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_index(index_path):
    df = pd.read_csv(index_path)
    embeddings = [np.fromstring(emb_str, sep=",", dtype=np.float32) for emb_str in df["clip_embedding"]]
    return df, torch.from_numpy(np.vstack(embeddings))


def encode_query(text, model, tokenizer, device):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding.squeeze(0)
    return (embedding / embedding.norm()).cpu()


def cosine_similarity(query_emb, index_embs):
    index_norm = index_embs / index_embs.norm(dim=1, keepdim=True)
    return torch.mv(index_norm, query_emb)


def main():
    args = parse_args()
    paths = ZeroShotPaths.from_env()
    index_csv = Path(args.index_csv) if args.index_csv else paths.retrieval_index
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found at {index_csv}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    clip_model = CLIPModel.from_pretrained(args.checkpoint_clip).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint_clip)
    clip_model.eval()

    df, embeddings = load_index(index_csv)
    scores = cosine_similarity(encode_query(args.query, clip_model, clip_tokenizer, device), embeddings)

    top_k = min(args.top_k, len(df))
    top_scores, top_indices = torch.topk(scores, top_k)
    for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        row = df.iloc[idx]
        print(f"{rank:>2}. score={score:.3f} modelId={row['modelId']} category={row['category']} caption='{row['caption']}' image={row['image_path']}")


if __name__ == "__main__":
    main()

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.dataset.split_dataset import DatasetSplitter
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Top-K similarity among clip embeddings using multi-view averaging")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--clip_emb_root", type=str, default="data/2D_render_clip_embeddings/tablechair_test")
    parser.add_argument("--result_dir", type=str, default=None)
    return parser.parse_args()

def load_all_views_clip_embeddings(model_ids, clip_emb_root):
    """
    Load 8-view clip embeddings per model_id.
    Returns tensor shape [N_models, 8, D].
    """
    all_embs = []
    for mid in model_ids:
        path = os.path.join(clip_emb_root, mid, "clip_embed.npy")
        emb8 = np.load(path)  # shape (8, D)
        # Normalize each view embedding
        emb8 = emb8 / np.linalg.norm(emb8, axis=1, keepdims=True)
        all_embs.append(emb8)
    all_embs = torch.tensor(np.stack(all_embs)).float()  # [N_models, 8, D]
    # import ipdb; ipdb.set_trace()
    return all_embs

def topk_accuracy_clip_vs_clipavg(dataloader, device, k, all_model_embs, model_id_to_idx):
    """
    Compute top-k accuracy by:
    - Comparing each sample's clip_emb with all 8 views of each model_id clip_emb.
    - Similarity(sample, model_id) = mean similarity across 8 views.
    """
    all_clip_embs = []
    all_sample_model_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Loading clip_emb for samples"):
            all_clip_embs.append(batch["clip_emb"])  # [B, D]
            all_sample_model_ids.extend(batch["model_id"])

    all_clip_embs = torch.cat(all_clip_embs, dim=0).to(device)  # [num_samples, D]
    num_samples = all_clip_embs.shape[0]

    num_models = all_model_embs.shape[0]

    # Map sample model_id to index in model embedding list
    labels = torch.tensor([model_id_to_idx[mid] for mid in all_sample_model_ids], device=device)  # [num_samples]

    # Compute similarity:
    # all_clip_embs: [num_samples, D]
    # all_model_embs: [num_models, 8, D]
    # Want similarity matrix: [num_samples, num_models]
    # similarity[i,j] = mean over 8 views of dot(all_clip_embs[i], all_model_embs[j, view])

    # Optional: normalize embeddings for cosine similarity
    # all_clip_embs = torch.nn.functional.normalize(all_clip_embs, dim=1)
    # all_model_embs = torch.nn.functional.normalize(all_model_embs, dim=2)

    # Compute dot products: broadcasting
    # [num_samples, 1, 1, D] * [1, num_models, 8, D] -> sum over last dim -> [num_samples, num_models, 8]
    dot_products = (all_clip_embs[:, None, None, :] * all_model_embs[None, :, :, :]).sum(dim=-1)  # [num_samples, num_models, 8]

    similarity = dot_products.mean(dim=-1)  # mean over 8 views -> [num_samples, num_models]

    # Get top-k indices per sample
    top_k_indices = torch.topk(similarity, k=k, dim=1).indices  # [num_samples, k]

    # Calculate how many samples have correct model_id in top-k
    correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
    acc = correct / num_samples

    return acc

if __name__ == "__main__":
    args = parse_args()

    result_dir = args.result_dir
    csv_path = "data/captions.tablechair.emb.csv"
    pcd_root = "data/text2shape_2048_pointcloud"
    batch_size = args.batch_size
    k = args.k
    num_samples = args.num_samples
    clip_emb_root = args.clip_emb_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = DatasetSplitter(csv_path)
    _, _, test_df = splitter.split_by_category()
    # print("Testing if test_df is consistent across runs...")
    # print(test_df.head())
    # sys.exit(0)
    test_dataset = PointCloudCLIPDataset(test_df, pcd_root, use_color=True)

    # Define subset_model_ids based on the first num_samples
    subset_model_ids = sorted(list(set(test_dataset.df["modelId"].values)))[:num_samples]

    # Take first num_samples samples only based on model_id order
    subset_indices = [test_dataset.df.index[test_dataset.df["modelId"] == mid][0] for mid in subset_model_ids]
    subset_dataset = Subset(test_dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

    # Unique model_ids in these samples (preserve order)
    model_id_to_idx = {mid: idx for idx, mid in enumerate(subset_model_ids)}
    # import ipdb; ipdb.set_trace()

    # Load all 8-view embeddings for model_ids
    all_model_embs = load_all_views_clip_embeddings(subset_model_ids, clip_emb_root).to(device)

    # Calculate top-k accuracy
    top_k_acc = topk_accuracy_clip_vs_clipavg(subset_loader, device, k, all_model_embs, model_id_to_idx)
    print(f"Top-{k} accuracy on first {num_samples} samples by multi-view clip embedding similarity: {top_k_acc:.4f}")

    # Save results
    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "top_k_clipavg_results.json"), "w") as f:
            json.dump({"top_k_accuracy": top_k_acc, "k": k, "num_samples": num_samples}, f, indent=4)
        print(f"Top-K results saved to {os.path.join(result_dir, 'top_k_clipavg_results.json')}")

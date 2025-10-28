import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.dataset.split_dataset import DatasetSplitter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_type", type=str, default="InfoNCE_duplicated_positive", help="Type of loss function to use")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_dir = f"results/point_cloud_contrastive/batch_size_{args.batch_size}/lr_{args.learning_rate}/temperature_{args.temperature}/"
    ids_path = os.path.join(results_dir, f"test_pointcloud_ids_{args.loss_type}.npy")
    embeddings_path = os.path.join(results_dir, f"test_pointcloud_embeddings_{args.loss_type}.npy")

    splitter = DatasetSplitter("data/captions.tablechair.emb.csv")
    _, _, test_df = splitter.split_by_category()
    test_dataset = PointCloudCLIPDataset(test_df, "data/text2shape_2048_pointcloud", use_color=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

    embeddings = np.load(embeddings_path, allow_pickle=True).item()  # Load embeddings as dictionary

    id_to_embedding = embeddings  # Dictionary of model_id -> embedding
    model_ids = list(id_to_embedding.keys())
    embedding_matrix = np.stack(list(id_to_embedding.values()))  # [N, D]

    correct = 0
    total = 0

    results = []

    for batch in tqdm(test_loader, desc="Calculating Top-K"):
        clip_emb = batch["clip_emb"].numpy()  # Convert to NumPy array
        clip_ids = batch["id"]  # List of IDs for the batch
        clip_model_ids = batch["model_id"]  # List of model IDs for the batch

        # Normalize clip embeddings
        clip_emb = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-10)

        for i, query_emb in enumerate(clip_emb):
            query_id = clip_ids[i]
            query_model_id = clip_model_ids[i]

            # Compute similarity
            similarities = embedding_matrix @ query_emb  # [N]

            # Get top-30 indices
            top_30_indices = np.argsort(similarities)[-30:][::-1]  # Indices of top-30 similarities
            top_30_model_ids = [model_ids[idx] for idx in top_30_indices]  # Get corresponding model IDs

            # Save result for this sample
            results.append({
                "id": str(query_id),  # Convert to string for JSON compatibility
                "model_id": str(query_model_id),  # Convert to string for JSON compatibility
                "top_30_model_ids": [str(mid) for mid in top_30_model_ids]  # Convert all model IDs to strings
            })

    # Save results to a JSON file
    output_path = os.path.join(results_dir, f"top_30_results_{args.loss_type}.json")
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved Top-30 results to {output_path}")



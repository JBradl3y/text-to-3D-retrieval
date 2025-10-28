import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print some samples from embeddings file")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the embeddings .npy file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to print")
    args = parser.parse_args()

    # Load embeddings
    embeddings = np.load(args.embeddings_path, allow_pickle=True).item()  # Load as dictionary

    # Print some samples
    print(f"Printing {args.num_samples} samples from {args.embeddings_path}:")
    for i, (model_id, embedding) in enumerate(embeddings.items()):
        if i >= args.num_samples:
            break
        norm = np.linalg.norm(embedding)
        print(f"Model ID: {model_id}, Embedding: {embedding[:5]}... (truncated), Norm: {norm:.4f}")  # Print model ID, first 5 values, and norm
        print(embedding.shape)

    print("Done.")
import pandas as pd
import torch
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm
import numpy as np
import os

# --- Paths ---
captions_file = "data/captions.tablechair.csv"
output_file = "data/captions.tablechair.emb.csv"

# --- Load CSV ---
df = pd.read_csv(captions_file)
df['category'] = df['category'].str.lower()

# 🔧 Keep only rows where 'description' is a string
df = df[df['description'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

# --- Load CLIP model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_name)

# --- Encode descriptions ---
max_length = 77
embeddings = []

print("Encoding descriptions with CLIP...")
for desc in tqdm(df['description'], total=len(df)):
    inputs = tokenizer(
        desc,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    embeddings.append(out.squeeze(0).cpu().numpy())  # save as numpy array

# --- Save embeddings in dataframe ---
df['clip_embedding'] = embeddings

# Optional: save numpy arrays as strings in CSV
df['clip_embedding'] = df['clip_embedding'].apply(lambda x: ','.join(map(str, x)))

# --- Save to new CSV ---
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
print(f"Saved CLIP embeddings CSV to {output_file}")

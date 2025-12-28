"""Push the mermaid dataset to Hugging Face Hub"""
import os
from datasets import load_from_disk
from huggingface_hub import login

# Get token from environment or prompt
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Please set HF_TOKEN environment variable or login interactively:")
    login()

# Load the dataset
ds = load_from_disk("datasets/diagrams_with_mermaid_codes")

print(f"Dataset loaded: {len(ds)} samples")
print(f"Columns: {ds.column_names}")

# Push to hub - change 'your-username/dataset-name' to your desired repo
REPO_NAME = "colinfrisch/diagrams_with_mermaid_codes"

if REPO_NAME:
    print(f"Pushing to {REPO_NAME}...")
    ds.push_to_hub(REPO_NAME, private=False)
    print(f"✅ Dataset pushed successfully to https://huggingface.co/datasets/{REPO_NAME}")
else:
    print("No repo name provided. Exiting.")


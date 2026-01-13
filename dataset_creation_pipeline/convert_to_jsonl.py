"""
Convert the diagrams_mermaid dataset to JSONL format for Mistral fine-tuning
with train and validation splits.

Mistral requires messages format: {"messages": [...]}
"""

import json
import os
import random

import pyarrow.ipc as ipc


def main():
    # Load the dataset from Arrow file
    arrow_path = "datasets/diagrams_mermaid/data-00000-of-00001.arrow"
    
    with open(arrow_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    
    # Convert to Python list of dicts
    data = table.to_pydict()
    samples = []
    
    for code, caption in zip(data["code"], data["caption"]):
        # Mistral fine-tuning format: messages array
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": caption
                },
                {
                    "role": "assistant",
                    "content": code
                }
            ]
        }
        samples.append(sample)
    
    print(f"Loaded {len(samples)} samples")
    
    # Shuffle and split into train (90%) and validation (10%)
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create output directory
    output_dir = "datasets/diagrams_mermaid/diagrams_mermaid_jsonl"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split as JSONL
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Saved train split to {train_path}")
    
    # Save validation split as JSONL
    val_path = os.path.join(output_dir, "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_data:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Saved validation split to {val_path}")
    
    # Print sample to verify format
    print("\nSample entry format:")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False)[:800] + "...")
    
    print(f"\nDone! JSONL files saved to {output_dir}/")

if __name__ == "__main__":
    main()

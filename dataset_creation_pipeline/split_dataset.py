#!/usr/bin/env python3
"""
Split the filtered dataset into train, validation, and test sets.
"""
import json
import random
from pathlib import Path


def split_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split a JSONL dataset into train, validation, and test sets.
    
    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save the split files
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all data
    print(f"Reading data from {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Shuffle the data
    print("Shuffling data...")
    random.shuffle(data)
    
    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size  # Remaining samples go to test
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size} ({train_size/total_samples*100:.1f}%)")
    print(f"  Validation: {val_size} ({val_size/total_samples*100:.1f}%)")
    print(f"  Test: {test_size} ({test_size/total_samples*100:.1f}%)")
    
    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.jsonl"
        print(f"\nWriting {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Saved {len(split_data)} samples")
    
    print("\nDataset split complete!")
    
    # Write metadata
    metadata_file = output_path / "split_metadata.json"
    metadata = {
        "total_samples": total_samples,
        "train_samples": train_size,
        "validation_samples": val_size,
        "test_samples": test_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "source_file": str(input_file)
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")


if __name__ == "__main__":
    # Configuration
    input_file = "datasets/diagrams_mermaid_filtered/filtered.jsonl"
    output_dir = "datasets/diagrams_mermaid_filtered/splits"
    
    # Split the dataset
    split_dataset(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

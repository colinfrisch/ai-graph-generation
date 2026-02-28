#!/usr/bin/env python3
"""
Absolute minimal Mistral fine-tuning example.

This script:
  1. Reads the Mermaid diagram train/validation splits
  2. Uploads them to Mistral Files API
  3. Creates a tiny 10-step fine-tuning job and starts it

It is intentionally minimal – no CLI, no rich progress UI.
"""

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mistralai import Mistral


# Reuse the same dataset location as in finetune.py
ROOT = Path(__file__).parent.parent
SPLITS_DIR = (
    ROOT
    / "dataset_creation_pipeline"
    / "datasets"
    / "diagrams_mermaid_filtered"
    / "splits"
)

TRAIN_FILE = SPLITS_DIR / "train.jsonl"
VAL_FILE = SPLITS_DIR / "validation.jsonl"


def upload_file(client: Mistral, path: Path):
    """Upload a JSONL file and return the resulting file object."""
    with open(path, "rb") as fh:
        return client.files.upload(
            file={
                "file_name": path.name,
                "content": fh,
            },
            purpose="fine-tune",
        )


def main() -> None:
    # Load API key from .env or environment
    load_dotenv(find_dotenv(usecwd=True) or ROOT / ".env", override=True)
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. "
            "Export it or add it to a .env file at the project root."
        )

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise SystemExit(
            f"Train/validation splits not found in {SPLITS_DIR}. "
            "Make sure you have generated train.jsonl and validation.jsonl."
        )

    client = Mistral(api_key=api_key)

    # Upload dataset files
    print(f"Uploading training file: {TRAIN_FILE}")
    train_file = upload_file(client, TRAIN_FILE)
    print(f" → training file_id={train_file.id}")

    print(f"Uploading validation file: {VAL_FILE}")
    val_file = upload_file(client, VAL_FILE)
    print(f" → validation file_id={val_file.id}")

    # Create a minimal 10-step fine-tuning job and start it immediately
    print("Creating fine-tuning job (10 steps, lr=1e-4)…")
    job = client.fine_tuning.jobs.create(
        model="open-mistral-7b",
        training_files=[{"file_id": train_file.id, "weight": 1}],
        validation_files=[val_file.id],
        hyperparameters={
            "training_steps": 10,
            "learning_rate": 1e-4,
        },
        auto_start=True,
    )

    print("Job created.")
    print(f"  job_id={job.id}")
    print(f"  status={job.status}")


if __name__ == "__main__":
    main()

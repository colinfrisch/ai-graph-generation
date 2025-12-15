"""
Dataset Exploration Script
Loads and explores the diagrams_with_captions dataset
"""

from datasets import load_from_disk

ds = load_from_disk("datasets/diagrams_with_captions")

print(ds)
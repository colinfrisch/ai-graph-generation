# Dataset Splits

This directory contains the train, validation, and test splits of the filtered Mermaid diagrams dataset.

## Split Details

- **Total samples**: 2,338
- **Split ratio**: 70% train / 15% validation / 15% test
- **Random seed**: 42 (for reproducibility)

## Files

| File | Samples | Percentage |
|------|---------|------------|
| `train.jsonl` | 1,636 | 70.0% |
| `validation.jsonl` | 350 | 15.0% |
| `test.jsonl` | 352 | 15.1% |
| `split_metadata.json` | - | Metadata |

## Data Format

Each file is in JSONL format (one JSON object per line) with the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Description of the diagram..."
    },
    {
      "role": "assistant",
      "content": "```mermaid\n...\n```"
    }
  ]
}
```

## Usage

### Python

```python
import json

# Load training data
train_data = []
with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))

# Access first sample
first_sample = train_data[0]
user_description = first_sample['messages'][0]['content']
mermaid_code = first_sample['messages'][1]['content']
```

### PyTorch/HuggingFace

```python
from datasets import load_dataset

# Load all splits
dataset = load_dataset('json', data_files={
    'train': 'train.jsonl',
    'validation': 'validation.jsonl',
    'test': 'test.jsonl'
})
```

## Notes

- The data was shuffled before splitting to ensure random distribution
- The random seed (42) ensures reproducible splits
- All samples from the original dataset are included in one of the three splits
- No data leakage between splits (each sample appears in exactly one split)

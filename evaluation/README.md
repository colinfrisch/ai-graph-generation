# Evaluation

Evaluates a custom fine-tuned Mistral model on the Mermaid diagram generation test set using the **Mistral Batch API** — all 352 test samples are processed in parallel in a single async job.

## How it works

```
test.jsonl
   │
   ▼
build batch JSONL          ← user prompts only (ground-truth stripped)
   │
   ▼
upload to Mistral          ← client.files.upload(purpose="batch")
   │
   ▼
submit batch job           ← client.batch.jobs.create(model=<your-ft-model>)
   │   Mistral runs all 352 requests in parallel (async)
   ▼
poll until SUCCESS         ← client.batch.jobs.get(job_id=...)
   │
   ▼
download results           ← client.files.download(file_id=job.output_file)
   │
   ▼
compute metrics            ← metrics.py
   │
   ▼
save JSON report           ← results/eval_<model>_<timestamp>.json
```

## Setup

```bash
cd evaluation
pip install -r requirements.txt
```

Set your Mistral API key (either export or in a `.env` file next to this script):

```bash
export MISTRAL_API_KEY=your_key_here
# or create a .env file:
echo "MISTRAL_API_KEY=your_key_here" > .env
```

## Usage

```bash
python evaluate.py --model ft:open-mistral-7b:your-org:20240430:xxxx
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Fine-tuned model ID (e.g. `ft:open-mistral-7b:org:date:id`) |
| `--test-file` | `../dataset_creation_pipeline/datasets/diagrams_mermaid_filtered/splits/test.jsonl` | Path to the test JSONL |
| `--output-dir` | `results/` | Directory for the evaluation report |
| `--max-tokens` | `1024` | Max tokens to generate per sample |
| `--poll-interval` | `10` | Seconds between batch-job status polls |

## Metrics

| Metric | Description |
|--------|-------------|
| **Valid Mermaid output** | % of responses that contain a recognisable ` ```mermaid ` block |
| **Exact match** | Strict character-for-character match of extracted Mermaid code |
| **Token-normalised exact match** | Same but whitespace-collapsed before comparing |
| **ROUGE-1** | Unigram overlap F1 between predicted and reference Mermaid code |
| **ROUGE-2** | Bigram overlap F1 |
| **ROUGE-L** | Longest common subsequence F1 |

## Output

The script produces a timestamped JSON report in `results/`:

```json
{
  "model": "ft:open-mistral-7b:...",
  "batch_job_id": "batch-xxxx",
  "timestamp": "20240501_143022",
  "total_samples": 352,
  "errors": 0,
  "aggregate_metrics": {
    "valid_mermaid": 0.95,
    "exact_match": 0.12,
    "token_exact_match": 0.14,
    "rouge1": 0.7831,
    "rouge2": 0.6412,
    "rougeL": 0.7213
  },
  "per_sample_results": [ ... ]
}
```

## Finding your fine-tuned model ID

After a successful fine-tuning job the model ID is available via the API:

```python
from mistralai import Mistral
client = Mistral(api_key="...")
job = client.fine_tuning.jobs.get(job_id="your-job-id")
print(job.fine_tuned_model)   # → ft:open-mistral-7b:org:YYYYMMDD:xxxx
```

Or navigate to [console.mistral.ai → Fine-tuned models](https://console.mistral.ai/build/finetuned-models).

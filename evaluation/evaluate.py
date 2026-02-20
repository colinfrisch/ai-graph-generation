#!/usr/bin/env python3
"""
Evaluate a custom fine-tuned Mistral model on the Mermaid diagram test set.

Workflow
--------
1. Load the test JSONL.
2. Build a batch-inference JSONL (user prompts only, ground-truth stripped).
3. Upload it to Mistral as a batch file (purpose="batch").
4. Submit a batch job → Mistral processes all samples in parallel (async).
5. Poll until the job reaches SUCCESS / FAILED.
6. Download results and compute metrics vs. ground truth.
7. Write a JSON report to results/.

Usage
-----
    python evaluate.py --model ft:open-mistral-7b:your-org:YYYYMMDD:xxxx

    # optional flags:
    #   --test-file  path to test.jsonl  (default: auto-detected)
    #   --output-dir path for reports    (default: results/)
    #   --max-tokens max tokens per response (default: 1024)
    #   --poll-interval seconds between status checks (default: 10)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mistralai import File, Mistral
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from metrics import aggregate_metrics, compute_sample_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

console = Console()

DEFAULT_TEST_FILE = (
    Path(__file__).parent.parent
    / "dataset_creation_pipeline"
    / "datasets"
    / "diagrams_mermaid_filtered"
    / "splits"
    / "test.jsonl"
)


def load_test_set(path: Path) -> list[dict]:
    """Load and return every sample from the JSONL test file."""
    samples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_batch_jsonl(samples: list[dict], max_tokens: int) -> bytes:
    """
    Build the JSONL content for the Mistral Batch API.

    Each line contains:
    - custom_id: the sample index (used later to match results to ground truth)
    - body.messages: only the *user* turn (no assistant ground truth)
    - body.max_tokens: token budget per response

    Returns bytes suitable for uploading.
    """
    buffer = BytesIO()
    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        request = {
            "custom_id": str(idx),
            "body": {
                "messages": user_messages,
                "max_tokens": max_tokens,
            },
        }
        buffer.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        buffer.write(b"\n")
    return buffer.getvalue()


def upload_batch_file(client: Mistral, content: bytes) -> str:
    """Upload the batch JSONL and return the file ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"eval_batch_{timestamp}.jsonl"
    console.print(f"[cyan]Uploading batch file[/cyan] ({len(content):,} bytes) …")
    uploaded = client.files.upload(
        file=File(file_name=file_name, content=content),
        purpose="batch",
    )
    console.print(f"[green]✓ Uploaded[/green] file_id={uploaded.id}")
    return uploaded.id


def submit_batch_job(client: Mistral, file_id: str, model: str) -> object:
    """Submit a batch job for chat completions and return the job object."""
    console.print(f"[cyan]Submitting batch job[/cyan] model={model} …")
    job = client.batch.jobs.create(
        input_files=[file_id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"purpose": "evaluation", "model": model},
    )
    console.print(f"[green]✓ Job created[/green] job_id={job.id}  status={job.status}")
    return job


def poll_until_done(client: Mistral, job_id: str, poll_interval: int) -> object:
    """
    Block until the batch job leaves QUEUED/RUNNING state.
    Prints live progress. Returns the final job object.
    """
    terminal_statuses = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Waiting for batch job …", total=None)

        while True:
            job = client.batch.jobs.get(job_id=job_id)
            total = job.total_requests or 1
            done = (job.succeeded_requests or 0) + (job.failed_requests or 0)
            pct = round(done / total * 100, 1)
            progress.update(
                task,
                description=(
                    f"[bold]{job.status}[/bold]  "
                    f"{done}/{total} requests  ({pct}%)"
                ),
            )
            if job.status in terminal_statuses:
                return job
            time.sleep(poll_interval)


def download_results(client: Mistral, job: object) -> list[dict]:
    """
    Download the output JSONL from a completed batch job.
    Returns a list of result dicts keyed by custom_id.
    """
    if job.status != "SUCCESS":
        console.print(
            f"[red]Batch job ended with status={job.status}. "
            "Attempting partial results …[/red]"
        )

    if not job.output_file:
        console.print("[red]No output file available.[/red]")
        return []

    console.print("[cyan]Downloading results …[/cyan]")
    stream = client.files.download(file_id=job.output_file)

    results = []
    for chunk in stream.stream:
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk
        for line in text.splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))

    console.print(f"[green]✓ Downloaded[/green] {len(results)} result lines")
    return results


def match_results_to_ground_truth(
    results: list[dict],
    samples: list[dict],
) -> list[dict]:
    """
    Align batch results (indexed by custom_id) with ground-truth samples.
    Returns a list of dicts with prediction + reference + metrics.
    """
    result_by_id: dict[str, dict] = {r["custom_id"]: r for r in results}
    evaluated = []

    for idx, sample in enumerate(samples):
        custom_id = str(idx)
        ref_messages = sample["messages"]
        reference = next(
            (m["content"] for m in ref_messages if m["role"] == "assistant"),
            "",
        )
        user_prompt = next(
            (m["content"] for m in ref_messages if m["role"] == "user"),
            "",
        )

        result = result_by_id.get(custom_id)
        if result is None or result.get("error"):
            prediction = ""
            error = result.get("error") if result else "missing"
        else:
            try:
                prediction = (
                    result["response"]["body"]["choices"][0]["message"]["content"]
                )
                error = None
            except (KeyError, IndexError, TypeError):
                prediction = ""
                error = "malformed_response"

        sample_metrics = compute_sample_metrics(prediction, reference)
        evaluated.append(
            {
                "custom_id": custom_id,
                "user_prompt": user_prompt,
                "reference": reference,
                "prediction": prediction,
                "error": error,
                **sample_metrics,
            }
        )

    return evaluated


def print_summary(agg: dict, total: int, errors: int) -> None:
    """Render a Rich table with the aggregated metrics."""
    table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")

    table.add_row("Samples", str(total))
    table.add_row("Errors / missing", str(errors))
    table.add_row("Valid Mermaid output", f"{agg.get('valid_mermaid', 0):.1%}")
    table.add_row("Exact match", f"{agg.get('exact_match', 0):.1%}")
    table.add_row("Token-normalised exact match", f"{agg.get('token_exact_match', 0):.1%}")
    table.add_row("ROUGE-1", f"{agg.get('rouge1', 0):.4f}")
    table.add_row("ROUGE-2", f"{agg.get('rouge2', 0):.4f}")
    table.add_row("ROUGE-L", f"{agg.get('rougeL', 0):.4f}")

    console.print(table)


def save_report(output_dir: Path, model: str, agg: dict, evaluated: list[dict], job_id: str) -> Path:
    """Write a JSON report to output_dir and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "-")
    report_path = output_dir / f"eval_{safe_model}_{timestamp}.json"

    report = {
        "model": model,
        "batch_job_id": job_id,
        "timestamp": timestamp,
        "total_samples": len(evaluated),
        "errors": sum(1 for e in evaluated if e["error"]),
        "aggregate_metrics": agg,
        "per_sample_results": evaluated,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv(find_dotenv(usecwd=True) or (Path(__file__).parent.parent / ".env"), override=True)

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Mistral model on the Mermaid test set via the Batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Fine-tuned model ID, e.g. ft:open-mistral-7b:your-org:20240430:xxxx  "
            "You can also pass any standard Mistral model ID."
        ),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="Path to the test JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory where the evaluation report will be saved.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between batch-job status polls.",
    )
    args = parser.parse_args()

    # ── API key ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        console.print(
            "[red]MISTRAL_API_KEY is not set.[/red] "
            "Export it or add it to a .env file next to this script."
        )
        sys.exit(1)

    # ── Load test set ────────────────────────────────────────────────────────
    if not args.test_file.exists():
        console.print(f"[red]Test file not found:[/red] {args.test_file}")
        sys.exit(1)

    console.rule("[bold]Loading test set")
    samples = load_test_set(args.test_file)
    console.print(f"Loaded [bold]{len(samples)}[/bold] samples from {args.test_file}")

    # ── Build & upload batch file ────────────────────────────────────────────
    console.rule("[bold]Preparing batch")
    batch_content = build_batch_jsonl(samples, max_tokens=args.max_tokens)
    client = Mistral(api_key=api_key)
    file_id = upload_batch_file(client, batch_content)

    # ── Submit job ───────────────────────────────────────────────────────────
    console.rule("[bold]Submitting batch job")
    job = submit_batch_job(client, file_id, args.model)

    # ── Poll until done ──────────────────────────────────────────────────────
    console.rule("[bold]Waiting for batch job (async, parallel execution)")
    job = poll_until_done(client, job.id, args.poll_interval)
    console.print(
        f"[green]✓ Job finished[/green]  "
        f"status={job.status}  "
        f"succeeded={job.succeeded_requests}  "
        f"failed={job.failed_requests}"
    )

    # ── Download & parse results ─────────────────────────────────────────────
    console.rule("[bold]Downloading results")
    raw_results = download_results(client, job)

    # ── Compute metrics ──────────────────────────────────────────────────────
    console.rule("[bold]Computing metrics")
    evaluated = match_results_to_ground_truth(raw_results, samples)
    agg = aggregate_metrics(
        [{k: v for k, v in e.items() if k not in ("custom_id", "user_prompt", "reference", "prediction", "error")}
         for e in evaluated]
    )
    errors = sum(1 for e in evaluated if e["error"])

    print_summary(agg, len(evaluated), errors)

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = save_report(args.output_dir, args.model, agg, evaluated, job.id)
    console.print(f"\n[bold green]Report saved →[/bold green] {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fine-tune ministral-3b-latest on the Mermaid diagram generation dataset
using the Mistral Fine-Tuning API.

Workflow
--------
1. Upload train.jsonl   → Mistral Files API  (purpose="fine-tune")
2. Upload validation.jsonl → same
3. Create a fine-tuning job (auto_start=False so we can inspect it first)
4. Wait for the job to reach VALIDATED status
5. Start the job
6. Poll until SUCCESS / FAILED, printing live progress
7. Print the fine-tuned model ID and save a JSON record to job_record.json

Usage
-----
    python finetune.py

    # with custom hyperparameters:
    python finetune.py --training-steps 200 --learning-rate 1e-4

    # skip re-uploading if you already have file IDs:
    python finetune.py --train-file-id <id> --val-file-id <id>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mistralai import Mistral
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
SPLITS_DIR = (
    ROOT
    / "dataset_creation_pipeline"
    / "datasets"
    / "diagrams_mermaid_filtered"
    / "splits"
)
DEFAULT_TRAIN_FILE = SPLITS_DIR / "train.jsonl"
DEFAULT_VAL_FILE = SPLITS_DIR / "validation.jsonl"
JOB_RECORD_PATH = Path(__file__).parent / "job_record.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "ministral-3b-latest"

# From the docs: Num epochs ≈ training_steps / training_file_size_in_MB
# Our training file is ~2 MB.  100 steps ≈ 50 epochs — too many.
# Rule of thumb for SFT on small datasets: aim for 3-5 epochs.
# With ~2 MB file: training_steps = target_epochs * file_MB = 3 * 2 ≈ 6 → very coarse.
# Mistral's own example uses 10 steps for demo purposes.
# We use 100 as a sensible default; adjust with --training-steps.
DEFAULT_TRAINING_STEPS = 100
DEFAULT_LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

def upload_file(client: Mistral, path: Path, label: str) -> str:
    """Upload a JSONL file for fine-tuning and return its file ID."""
    size_kb = path.stat().st_size / 1024
    console.print(f"[cyan]Uploading {label}[/cyan]  ({size_kb:,.1f} KB)  {path.name} …")
    with open(path, "rb") as fh:
        uploaded = client.files.upload(
            file={
                "file_name": path.name,
                "content": fh,
            },
            purpose="fine-tune",
        )
    console.print(f"[green]✓ Uploaded {label}[/green]  file_id={uploaded.id}")
    return uploaded.id


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def create_job(
    client: Mistral,
    train_file_id: str,
    val_file_id: str,
    training_steps: int,
    learning_rate: float,
    suffix: str | None,
) -> object:
    """Create a fine-tuning job with auto_start=False."""
    console.print(
        f"\n[cyan]Creating fine-tuning job[/cyan]  "
        f"model={MODEL}  steps={training_steps}  lr={learning_rate}"
    )

    kwargs = dict(
        model=MODEL,
        training_files=[{"file_id": train_file_id, "weight": 1}],
        validation_files=[val_file_id],
        hyperparameters={
            "training_steps": training_steps,
            "learning_rate": learning_rate,
        },
        auto_start=False,
    )
    if suffix:
        kwargs["suffix"] = suffix

    try:
        job = client.fine_tuning.jobs.create(**kwargs)
    except Exception as e:
        err_str = str(e)
        if "not available for this type of fine-tuning" in err_str or "Available model(s)" in err_str:
            console.print(
                "\n[bold red]Fine-tuning blocked by Mistral platform.[/bold red]\n\n"
                "The API returned: [italic]'Model not available for this type of fine-tuning "
                "(completion). Available model(s): '[/italic]\n\n"
                "This is an [bold]account-level feature restriction[/bold], not a credits issue.\n"
                "Fine-tuning may not be enabled for your current account tier.\n\n"
                "Options to resolve:\n"
                "  1. [bold]Try the web UI[/bold]: https://console.mistral.ai/build/finetuned-models\n"
                "     → 'New fine-tuning job' → select the already-uploaded files\n\n"
                "  2. [bold]Contact Mistral support[/bold]: support@mistral.ai\n"
                "     → Ask to enable fine-tuning (completion/SFT) on your account\n\n"
                "  3. Once access is granted, re-run with your pre-uploaded files:\n\n"
                f"     [cyan]python finetune.py \\\\\n"
                f"       --train-file-id {train_file_id} \\\\\n"
                f"       --val-file-id   {val_file_id} \\\\\n"
                f"       --training-steps {training_steps} \\\\\n"
                f"       --learning-rate  {learning_rate}[/cyan]\n"
            )
            sys.exit(1)
        raise
    console.print(f"[green]✓ Job created[/green]  job_id={job.id}  status={job.status}")
    return job


def wait_for_validated(client: Mistral, job_id: str, poll_interval: int) -> object:
    """
    Poll until the job leaves QUEUED and reaches VALIDATED (or a terminal state).
    Returns the latest job object.
    """
    validating_statuses = {"QUEUED"}
    terminal_statuses = {"VALIDATED", "FAILED", "CANCELLED"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Waiting for job validation …", total=None)
        while True:
            job = client.fine_tuning.jobs.get(job_id=job_id)
            progress.update(task, description=f"Status: [bold]{job.status}[/bold]")
            if job.status in terminal_statuses:
                return job
            if job.status not in validating_statuses:
                # May reach RUNNING if auto_start was flipped server-side
                return job
            time.sleep(poll_interval)


def start_job(client: Mistral, job_id: str) -> object:
    """Start the validated fine-tuning job."""
    console.print(f"[cyan]Starting job[/cyan]  job_id={job_id} …")
    client.fine_tuning.jobs.start(job_id=job_id)
    job = client.fine_tuning.jobs.get(job_id=job_id)
    console.print(f"[green]✓ Job started[/green]  status={job.status}")
    return job


def poll_until_done(client: Mistral, job_id: str, poll_interval: int) -> object:
    """
    Poll the job until it reaches a terminal state (SUCCESS / FAILED / etc.).
    Prints a live status line including progress percentages when available.
    """
    terminal_statuses = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Training …", total=None)
        while True:
            job = client.fine_tuning.jobs.get(job_id=job_id)

            # Build a descriptive status string
            status_str = f"[bold]{job.status}[/bold]"

            # Some job objects expose training events with loss info
            if hasattr(job, "events") and job.events:
                latest = job.events[-1]
                if hasattr(latest, "data") and latest.data:
                    data = latest.data
                    parts = []
                    if hasattr(data, "train_loss") and data.train_loss is not None:
                        parts.append(f"train_loss={data.train_loss:.4f}")
                    if hasattr(data, "valid_loss") and data.valid_loss is not None:
                        parts.append(f"val_loss={data.valid_loss:.4f}")
                    if parts:
                        status_str += "  " + "  ".join(parts)

            progress.update(task, description=status_str)

            if job.status in terminal_statuses:
                return job
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_job_summary(job: object) -> None:
    table = Table(title="Fine-Tuning Job Summary", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Job ID", str(job.id))
    table.add_row("Status", str(job.status))
    table.add_row("Model", str(getattr(job, "model", "—")))
    table.add_row(
        "Fine-tuned model ID",
        f"[bold green]{job.fine_tuned_model}[/bold green]"
        if getattr(job, "fine_tuned_model", None)
        else "[dim]not yet available[/dim]",
    )
    if getattr(job, "created_at", None) and getattr(job, "completed_at", None):
        duration = job.completed_at - job.created_at
        table.add_row("Duration", f"{duration}s")

    console.print(table)


def save_job_record(job: object, train_file_id: str, val_file_id: str) -> None:
    record = {
        "job_id": str(job.id),
        "status": str(job.status),
        "model": str(getattr(job, "model", MODEL)),
        "fine_tuned_model": str(getattr(job, "fine_tuned_model", None)),
        "train_file_id": train_file_id,
        "val_file_id": val_file_id,
    }
    with open(JOB_RECORD_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    console.print(f"[bold green]Job record saved →[/bold green] {JOB_RECORD_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(find_dotenv(usecwd=True) or ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(
        description=f"Fine-tune {MODEL} on the Mermaid diagram dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help=(
            "Number of training steps.  "
            "Rule of thumb: steps ≈ target_epochs × training_file_MB.  "
            "Our training file is ~2 MB, so 100 steps ≈ 50 gradient updates per MB."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Peak learning rate (cosine decay with linear warmup).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the fine-tuned model name.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=DEFAULT_VAL_FILE,
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--train-file-id",
        type=str,
        default=None,
        help="Reuse a previously uploaded training file ID (skips upload).",
    )
    parser.add_argument(
        "--val-file-id",
        type=str,
        default=None,
        help="Reuse a previously uploaded validation file ID (skips upload).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between job status polls.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Submit the job and exit immediately without waiting for it to finish.  "
            "The job ID is saved to job_record.json so you can resume monitoring later."
        ),
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

    client = Mistral(api_key=api_key)

    # ── Upload files ─────────────────────────────────────────────────────────
    console.rule("[bold]Step 1 — Upload dataset files")

    if args.train_file_id:
        train_file_id = args.train_file_id
        console.print(f"[yellow]Reusing training file[/yellow]  id={train_file_id}")
    else:
        if not args.train_file.exists():
            console.print(f"[red]Training file not found:[/red] {args.train_file}")
            sys.exit(1)
        train_file_id = upload_file(client, args.train_file, "training file")

    if args.val_file_id:
        val_file_id = args.val_file_id
        console.print(f"[yellow]Reusing validation file[/yellow]  id={val_file_id}")
    else:
        if not args.val_file.exists():
            console.print(f"[red]Validation file not found:[/red] {args.val_file}")
            sys.exit(1)
        val_file_id = upload_file(client, args.val_file, "validation file")

    # ── Create job ───────────────────────────────────────────────────────────
    console.rule("[bold]Step 2 — Create fine-tuning job")
    job = create_job(
        client,
        train_file_id=train_file_id,
        val_file_id=val_file_id,
        training_steps=args.training_steps,
        learning_rate=args.learning_rate,
        suffix=args.suffix,
    )

    # ── Wait for VALIDATED ───────────────────────────────────────────────────
    console.rule("[bold]Step 3 — Wait for validation")
    if job.status not in {"VALIDATED", "RUNNING", "SUCCESS"}:
        job = wait_for_validated(client, job.id, args.poll_interval)

    if job.status == "FAILED":
        console.print(f"[red]Job failed during validation.[/red]  job_id={job.id}")
        save_job_record(job, train_file_id, val_file_id)
        sys.exit(1)

    # ── Start job ────────────────────────────────────────────────────────────
    console.rule("[bold]Step 4 — Start training")
    if job.status == "VALIDATED":
        job = start_job(client, job.id)
    else:
        console.print(f"Job already in status=[bold]{job.status}[/bold], skipping manual start.")

    # ── Optionally exit early ────────────────────────────────────────────────
    if args.no_wait:
        console.print(
            f"\n[yellow]--no-wait flag set.[/yellow] "
            f"Job [bold]{job.id}[/bold] is running.  "
            "Monitor it at https://console.mistral.ai/build/finetuned-models"
        )
        save_job_record(job, train_file_id, val_file_id)
        return

    # ── Poll until done ──────────────────────────────────────────────────────
    console.rule("[bold]Step 5 — Monitor training (async polling)")
    job = poll_until_done(client, job.id, args.poll_interval)

    # ── Summary ──────────────────────────────────────────────────────────────
    console.rule("[bold]Results")
    print_job_summary(job)

    if job.status == "SUCCESS" and getattr(job, "fine_tuned_model", None):
        console.print(
            f"\n[bold green]Fine-tuned model is ready![/bold green]\n"
            f"Model ID: [bold]{job.fine_tuned_model}[/bold]\n\n"
            "Use it in evaluation:\n"
            f"  python ../evaluation/evaluate.py --model {job.fine_tuned_model}"
        )
    elif job.status != "SUCCESS":
        console.print(f"[red]Training ended with status={job.status}[/red]")

    save_job_record(job, train_file_id, val_file_id)


if __name__ == "__main__":
    main()

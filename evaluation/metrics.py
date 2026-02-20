"""
Evaluation metrics for Mermaid diagram generation.
"""
import re
from typing import Optional

from rouge_score import rouge_scorer


def extract_mermaid_code(text: str) -> Optional[str]:
    """
    Extract the raw Mermaid code from a markdown code block.
    Returns the code inside ```mermaid ... ``` or None if not found.
    """
    pattern = r"```mermaid\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # Fallback: if the text itself starts with a known mermaid keyword, treat it as raw code
    known_keywords = (
        "graph", "flowchart", "sequenceDiagram", "classDiagram",
        "stateDiagram", "erDiagram", "gantt", "pie", "mindmap",
        "timeline", "gitGraph", "xychart-beta", "block-beta", "quadrantChart",
    )
    stripped = text.strip()
    if any(stripped.startswith(kw) for kw in known_keywords):
        return stripped
    return None


def is_valid_mermaid(text: str) -> bool:
    """
    Heuristic check: returns True if the output contains a recognisable
    Mermaid diagram block.
    """
    return extract_mermaid_code(text) is not None


def exact_match(prediction: str, reference: str) -> bool:
    """
    Exact string match after normalising whitespace.
    Compares the extracted Mermaid code only (ignoring the markdown fence).
    """
    pred_code = extract_mermaid_code(prediction) or prediction.strip()
    ref_code = extract_mermaid_code(reference) or reference.strip()
    return pred_code == ref_code


def normalise_mermaid(code: str) -> str:
    """Collapse runs of whitespace for a more lenient comparison."""
    return re.sub(r"\s+", " ", code).strip()


def token_exact_match(prediction: str, reference: str) -> bool:
    """Exact match after whitespace normalisation."""
    pred_code = extract_mermaid_code(prediction) or prediction.strip()
    ref_code = extract_mermaid_code(reference) or reference.strip()
    return normalise_mermaid(pred_code) == normalise_mermaid(ref_code)


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores between the extracted
    Mermaid code blocks of prediction and reference.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    pred_code = extract_mermaid_code(prediction) or prediction.strip()
    ref_code = extract_mermaid_code(reference) or reference.strip()
    scores = scorer.score(ref_code, pred_code)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_sample_metrics(prediction: str, reference: str) -> dict:
    """
    Compute all metrics for a single prediction/reference pair.
    """
    valid = is_valid_mermaid(prediction)
    em = exact_match(prediction, reference)
    tem = token_exact_match(prediction, reference)
    rouge = rouge_scores(prediction, reference) if valid else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {
        "valid_mermaid": valid,
        "exact_match": em,
        "token_exact_match": tem,
        **rouge,
    }


def aggregate_metrics(sample_metrics: list[dict]) -> dict:
    """
    Aggregate per-sample metrics into dataset-level averages.
    """
    n = len(sample_metrics)
    if n == 0:
        return {}

    keys = ["valid_mermaid", "exact_match", "token_exact_match", "rouge1", "rouge2", "rougeL"]
    return {
        key: round(sum(m[key] for m in sample_metrics) / n, 4)
        for key in keys
    }

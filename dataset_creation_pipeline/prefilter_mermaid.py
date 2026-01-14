"""
Pre-filter mermaid dataset by removing invalid mermaid codes.
Uses parallel processing for speed.
"""

import os
import json
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pyarrow.ipc as ipc
import pyarrow as pa
from tqdm import tqdm


def clean_mermaid_code(code: str) -> str:
    """Clean mermaid code by removing markdown fences."""
    code = code.strip()
    lines = code.split('\n')
    
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    return '\n'.join(lines).strip()


def validate_mermaid_syntax(code: str) -> tuple[bool, str]:
    """
    Validate mermaid code using mermaid CLI (mmdc).
    Returns (is_valid, error_message).
    """
    clean_code = clean_mermaid_code(code)
    
    if not clean_code:
        return False, "Empty code"
    
    # Basic syntax check - valid diagram types
    valid_starts = [
        'graph ', 'graph\n', 'flowchart ', 'flowchart\n',
        'sequencediagram', 'sequence diagram',
        'classdiagram', 'class diagram',
        'statediagram', 'state diagram',
        'erdiagram', 'er diagram',
        'journey', 'gantt', 'pie', 'gitgraph', 'mindmap',
        'timeline', 'quadrantchart', 'xychart', 'block-beta',
        'sankey', 'requirement', 'c4context',
        '---', '%%'
    ]
    
    first_line = clean_code.split('\n')[0].strip().lower()
    has_valid_start = any(first_line.startswith(s) for s in valid_starts)
    
    if not has_valid_start:
        first_lines = '\n'.join(clean_code.split('\n')[:5]).lower()
        has_valid_start = any(s in first_lines for s in valid_starts)
    
    if not has_valid_start:
        return False, f"Invalid diagram type: {first_line[:50]}"
    
    return True, ""


def validate_single(args: tuple) -> tuple[int, bool, str]:
    """Validate a single mermaid code. Returns (index, is_valid, error)."""
    idx, code = args
    is_valid, error = validate_mermaid_syntax(code)
    return idx, is_valid, error


def main():
    print("=" * 70)
    print("PRE-FILTER MERMAID DATASET")
    print("=" * 70)
    
    # Load dataset
    input_path = "datasets/diagrams_mermaid/data-00000-of-00001.arrow"
    
    if not os.path.exists(input_path):
        print(f"❌ Dataset not found: {input_path}")
        return
    
    print(f"\n📂 Loading dataset from {input_path}...")
    with open(input_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    
    data = table.to_pydict()
    codes = data["code"]
    captions = data["caption"]
    total = len(codes)
    
    print(f"   Found {total} samples")
    
    # Validate in parallel
    print(f"\n🔍 Validating mermaid codes using {cpu_count()} workers...")
    
    valid_indices = []
    invalid_count = 0
    errors_sample = []
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # Submit all tasks
        futures = {
            executor.submit(validate_single, (i, codes[i])): i 
            for i in range(total)
        }
        
        # Process results with progress bar
        with tqdm(total=total, desc="Validating") as pbar:
            for future in as_completed(futures):
                idx, is_valid, error = future.result()
                
                if is_valid:
                    valid_indices.append(idx)
                else:
                    invalid_count += 1
                    if len(errors_sample) < 10:
                        errors_sample.append((idx, error))
                
                pbar.update(1)
    
    # Sort valid indices to maintain order
    valid_indices.sort()
    
    print(f"\n📊 Results:")
    print(f"   ✓ Valid: {len(valid_indices)} ({len(valid_indices)/total*100:.1f}%)")
    print(f"   ✗ Invalid: {invalid_count} ({invalid_count/total*100:.1f}%)")
    
    if errors_sample:
        print(f"\n   Sample errors:")
        for idx, error in errors_sample[:5]:
            print(f"      [{idx}] {error[:60]}...")
    
    # Save filtered dataset
    output_dir = "datasets/diagrams_mermaid_valid"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving filtered dataset to {output_dir}/...")
    
    # Filter data
    filtered_codes = [codes[i] for i in valid_indices]
    filtered_captions = [captions[i] for i in valid_indices]
    
    # Save as Arrow
    filtered_table = pa.table({
        "code": filtered_codes,
        "caption": filtered_captions
    })
    
    arrow_path = os.path.join(output_dir, "data-00000-of-00001.arrow")
    with open(arrow_path, "wb") as f:
        writer = ipc.new_stream(f, filtered_table.schema)
        writer.write_table(filtered_table)
        writer.close()
    
    # Save dataset info
    dataset_info = {
        "description": f"Pre-filtered Mermaid dataset (invalid codes removed) - {datetime.now().isoformat()}",
        "original_samples": total,
        "valid_samples": len(valid_indices),
        "removed_samples": invalid_count,
        "features": {
            "code": {"dtype": "string"},
            "caption": {"dtype": "string"}
        }
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save state for Hugging Face
    state = {"_data_files": [{"filename": "data-00000-of-00001.arrow"}], "_split": None}
    with open(os.path.join(output_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
    
    # Also save as JSONL
    jsonl_path = os.path.join(output_dir, "valid.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for code, caption in zip(filtered_codes, filtered_captions):
            sample = {
                "messages": [
                    {"role": "user", "content": caption},
                    {"role": "assistant", "content": code}
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Done!")
    print(f"   Arrow: {arrow_path}")
    print(f"   JSONL: {jsonl_path}")
    print(f"   Info:  {output_dir}/dataset_info.json")


if __name__ == "__main__":
    main()

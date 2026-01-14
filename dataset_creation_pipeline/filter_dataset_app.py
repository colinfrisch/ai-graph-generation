"""
Tinder-style Streamlit app for filtering the Mermaid dataset.
Swipe right (or press →) to keep, swipe left (or press ←) to discard.
"""

import os
import json
import html
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pyarrow.ipc as ipc
import pyarrow as pa

# Page config
st.set_page_config(
    page_title="Dataset Filter",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def clean_mermaid_code(code: str) -> str:
    """Clean mermaid code by removing markdown fences and extra whitespace."""
    code = code.strip()
    
    # Remove markdown code fences if present
    lines = code.split('\n')
    
    # Check if first line is a code fence
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    
    # Check if last line is a code fence
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    return '\n'.join(lines).strip()


def validate_mermaid_basic(code: str) -> tuple[bool, str]:
    """
    Basic validation of mermaid code syntax.
    Returns (is_valid, error_message).
    """
    clean_code = clean_mermaid_code(code)
    
    if not clean_code:
        return False, "Empty code"
    
    # Valid mermaid diagram types
    valid_starts = [
        'graph ', 'graph\n', 'flowchart ', 'flowchart\n',
        'sequencediagram', 'sequence diagram',
        'classdiagram', 'class diagram',
        'statediagram', 'state diagram',
        'erdiagram', 'er diagram', 'entityrelationshipdiagram',
        'journey', 'gantt', 'pie', 'gitgraph', 'mindmap',
        'timeline', 'quadrantchart', 'xychart', 'block-beta',
        'sankey', 'requirement', 'c4context', 'c4container', 'c4component', 'c4deployment',
        '---', '%%'  # YAML frontmatter or comments at start
    ]
    
    first_line = clean_code.split('\n')[0].strip().lower()
    
    # Check if starts with valid diagram type
    has_valid_start = any(first_line.startswith(s) for s in valid_starts)
    
    if not has_valid_start:
        # Check if it might be a valid type anywhere in first few lines
        first_lines = '\n'.join(clean_code.split('\n')[:5]).lower()
        has_valid_start = any(s in first_lines for s in valid_starts)
    
    if not has_valid_start:
        return False, f"Invalid diagram type. First line: {first_line[:50]}"
    
    return True, ""


@st.cache_data
def validate_all_codes(codes: list) -> dict:
    """Pre-validate all mermaid codes and return invalid indices."""
    invalid_indices = {}
    for i, code in enumerate(codes):
        is_valid, error = validate_mermaid_basic(code)
        if not is_valid:
            invalid_indices[i] = error
    return invalid_indices


def render_mermaid(code: str, height: int = 400) -> None:
    """Render Mermaid diagram using Mermaid.js"""
    # Clean the code first
    clean_code = clean_mermaid_code(code)
    
    # Encode the code as base64 to avoid any escaping issues
    import base64
    code_b64 = base64.b64encode(clean_code.encode('utf-8')).decode('utf-8')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                background: #fff;
                width: 100%;
                height: {height}px;
                overflow: auto;
            }}
            #mermaid-container {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 8px;
            }}
            #mermaid-container svg {{
                max-width: 100%;
                max-height: 100%;
                width: auto;
                height: auto;
            }}
            .error {{
                color: #ff6b6b;
                font-family: monospace;
                padding: 12px;
                background: #fff5f5;
                border-left: 3px solid #ff6b6b;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div id="mermaid-container">
            <pre class="mermaid" id="diagram"></pre>
        </div>
        <script>
            // Decode the base64 code
            const codeB64 = "{code_b64}";
            const code = atob(codeB64);
            
            // Set the code content
            document.getElementById('diagram').textContent = code;
            
            // Initialize mermaid with error handling
            mermaid.initialize({{ 
                startOnLoad: false,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{ htmlLabels: true, curve: 'basis' }},
                logLevel: 'error'
            }});
            
            // Render with proper error handling
            async function renderDiagram() {{
                try {{
                    const {{ svg }} = await mermaid.render('mermaid-svg', code);
                    document.getElementById('mermaid-container').innerHTML = svg;
                }} catch (err) {{
                    let errorMsg = 'Unknown error';
                    if (err && err.message) {{
                        errorMsg = err.message;
                    }} else if (typeof err === 'string') {{
                        errorMsg = err;
                    }} else if (err && err.str) {{
                        errorMsg = err.str;
                    }}
                    document.getElementById('mermaid-container').innerHTML = 
                        '<div class="error"><strong>⚠️ Mermaid Syntax Error:</strong><br>' + errorMsg + '</div>';
                }}
            }}
            
            renderDiagram();
        </script>
    </body>
    </html>
    """
    
    components.html(html_content, height=height, scrolling=True)

# Compact CSS
st.markdown("""
<style>
    .stApp { background: #1e1e2e; }
    .block-container { padding-top: 2rem !important; padding-bottom: 0 !important; }
    
    .header-row { 
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        margin-bottom: 0.5rem; 
        margin-top: 0.5rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid #313244;
    }
    .title { 
        color: #cdd6f4; 
        font-size: 1.3rem; 
        font-weight: 600; 
        margin: 0; 
    }
    .counter { 
        color: #a6adc8; 
        font-size: 0.95rem; 
    }
    
    .stats { 
        display: flex; 
        gap: 0.8rem; 
        align-items: center;
    }
    .stat { 
        padding: 0.35rem 0.7rem; 
        border-radius: 4px; 
        font-size: 0.85rem; 
        font-weight: 500;
    }
    .stat-keep { background: #1e3a29; color: #a6e3a1; }
    .stat-discard { background: #3a1e1e; color: #f38ba8; }
    .stat-remaining { background: #1e2d3a; color: #89b4fa; }
    
    .instruction-box {
        background: #313244;
        border-radius: 8px;
        padding: 0.8rem;
        color: #cdd6f4;
        font-size: 0.95rem;
        line-height: 1.4;
        height: 100%;
        overflow: auto;
    }
    .instruction-label { color: #f9e2af; font-size: 0.8rem; margin-bottom: 0.4rem; font-weight: 600; }
    
    .btn-row { display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem; }
    div.stButton > button { font-weight: 600; padding: 0.5rem 2rem; border-radius: 6px; }
    
    .keyboard-hint { text-align: center; color: #6c7086; font-size: 0.75rem; margin-top: 0.3rem; }
    .keyboard-hint kbd { background: #313244; padding: 0.2rem 0.4rem; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    """Load the Mermaid dataset from Arrow file."""
    arrow_path = "datasets/diagrams_mermaid/data-00000-of-00001.arrow"
    
    if not os.path.exists(arrow_path):
        st.error(f"Dataset not found at {arrow_path}")
        return None, None
    
    with open(arrow_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    
    data = table.to_pydict()
    return data["code"], data["caption"]


def save_filtered_dataset(kept_indices, codes, captions, current_index=None, discarded_count=None):
    """Save the filtered dataset and optionally save progress for resuming."""
    output_dir = "datasets/diagrams_mermaid_filtered"
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter the data
    filtered_codes = [codes[i] for i in kept_indices]
    filtered_captions = [captions[i] for i in kept_indices]
    
    # Save as Arrow
    table = pa.table({
        "code": filtered_codes,
        "caption": filtered_captions
    })
    
    arrow_path = os.path.join(output_dir, "data-00000-of-00001.arrow")
    with open(arrow_path, "wb") as f:
        writer = ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()
    
    # Save dataset info
    dataset_info = {
        "description": f"Manually filtered Mermaid dataset - {datetime.now().isoformat()}",
        "num_samples": len(kept_indices),
        "features": {
            "code": {"dtype": "string"},
            "caption": {"dtype": "string"}
        }
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Also save as JSONL for convenience
    jsonl_path = os.path.join(output_dir, "filtered.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for code, caption in zip(filtered_codes, filtered_captions):
            sample = {
                "messages": [
                    {"role": "user", "content": caption},
                    {"role": "assistant", "content": code}
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Save session state for resuming (if provided)
    if current_index is not None:
        progress_path = os.path.join(output_dir, "progress.json")
        progress = {
            "current_index": current_index,
            "kept_indices": kept_indices,
            "discarded_count": discarded_count or 0,
            "saved_at": datetime.now().isoformat()
        }
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
    
    return output_dir, len(kept_indices)


def load_progress(codes, captions):
    """Load saved progress. If no progress.json exists, reconstruct from filtered dataset."""
    output_dir = "datasets/diagrams_mermaid_filtered"
    progress_path = os.path.join(output_dir, "progress.json")
    
    # First try to load explicit progress file
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    
    # No progress file - try to reconstruct from filtered dataset
    arrow_path = os.path.join(output_dir, "data-00000-of-00001.arrow")
    if os.path.exists(arrow_path):
        # Load the filtered dataset to find which indices were kept
        with open(arrow_path, "rb") as f:
            reader = ipc.open_stream(f)
            table = reader.read_all()
        
        filtered_data = table.to_pydict()
        filtered_codes = filtered_data.get("code", [])
        
        if filtered_codes:
            # Find the indices in the original dataset that match filtered items
            kept_indices = []
            for fc in filtered_codes:
                for i, orig_code in enumerate(codes):
                    if orig_code == fc and i not in kept_indices:
                        kept_indices.append(i)
                        break
            
            if kept_indices:
                # The next index to process is after the last kept item
                last_processed = max(kept_indices) + 1
                return {
                    "current_index": last_processed,
                    "kept_indices": kept_indices,
                    "discarded_count": last_processed - len(kept_indices),  # Estimate
                    "saved_at": "reconstructed from filtered dataset"
                }
    
    return None


def main():
    # Load dataset
    codes, captions = load_dataset()
    if codes is None:
        return
    
    total_samples = len(codes)
    
    # Pre-validate all codes to find invalid ones
    invalid_codes = validate_all_codes(tuple(codes))  # tuple for caching
    
    # Check for saved progress (pass codes/captions to reconstruct if needed)
    saved_progress = load_progress(codes, captions)
    
    # Initialize session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "kept_indices" not in st.session_state:
        st.session_state.kept_indices = []
    if "discarded_count" not in st.session_state:
        st.session_state.discarded_count = 0
    if "finished" not in st.session_state:
        st.session_state.finished = False
    if "asked_resume" not in st.session_state:
        st.session_state.asked_resume = False
    if "auto_skip_invalid" not in st.session_state:
        st.session_state.auto_skip_invalid = True
    if "auto_discarded" not in st.session_state:
        st.session_state.auto_discarded = 0
    
    # Offer to resume if saved progress exists
    if saved_progress and not st.session_state.asked_resume and st.session_state.current_index == 0:
        saved_idx = saved_progress.get("current_index", 0)
        saved_kept = len(saved_progress.get("kept_indices", []))
        saved_discarded = saved_progress.get("discarded_count", 0)
        
        st.markdown(f"""
            <div style="background:#313244; padding:1rem; border-radius:8px; margin-bottom:1rem; color:#cdd6f4;">
                <b>📌 Saved progress found!</b><br>
                Position: {saved_idx}/{total_samples} | Kept: {saved_kept} | Discarded: {saved_discarded}
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Resume", use_container_width=True):
                st.session_state.current_index = saved_progress["current_index"]
                st.session_state.kept_indices = saved_progress["kept_indices"]
                st.session_state.discarded_count = saved_progress["discarded_count"]
                st.session_state.asked_resume = True
                st.rerun()
        with c2:
            if st.button("🆕 Start Fresh", use_container_width=True):
                st.session_state.asked_resume = True
                st.rerun()
        return
    
    # Check if done
    if st.session_state.finished or st.session_state.current_index >= total_samples:
        st.session_state.finished = True
        kept = len(st.session_state.kept_indices)
        discarded = st.session_state.discarded_count
        
        auto_skipped = st.session_state.get("auto_discarded", 0)
        auto_text = f" (auto: {auto_skipped})" if auto_skipped > 0 else ""
        st.markdown(f"""
            <div style="text-align:center; padding:2rem; color:#cdd6f4;">
                <h2>✨ Done!</h2>
                <p><span class="stat stat-keep">✓ Kept: {kept}</span> 
                   <span class="stat stat-discard">✗ Discarded: {discarded}{auto_text}</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("💾 Save Final Dataset", use_container_width=True):
                output_dir, num_saved = save_filtered_dataset(
                    st.session_state.kept_indices, codes, captions,
                    st.session_state.current_index, st.session_state.discarded_count
                )
                st.success(f"Saved {num_saved} samples to `{output_dir}/`")
            if st.button("🔄 Restart", use_container_width=True):
                st.session_state.current_index = 0
                st.session_state.kept_indices = []
                st.session_state.discarded_count = 0
                st.session_state.finished = False
                st.session_state.asked_resume = False
                st.session_state.auto_discarded = 0
                st.rerun()
        return
    
    current = st.session_state.current_index
    kept = len(st.session_state.kept_indices)
    discarded = st.session_state.discarded_count
    
    # Auto-skip invalid mermaid codes
    if st.session_state.auto_skip_invalid:
        while current < total_samples and current in invalid_codes:
            st.session_state.discarded_count += 1
            st.session_state.auto_discarded += 1
            st.session_state.current_index += 1
            current = st.session_state.current_index
            discarded = st.session_state.discarded_count
        
        # Check if we've reached the end after auto-skipping
        if current >= total_samples:
            st.session_state.finished = True
            st.rerun()
    
    # Compact header: title + stats + counter (single line)
    auto_skipped_text = f" (auto: {st.session_state.auto_discarded})" if st.session_state.auto_discarded > 0 else ""
    st.markdown(f"""
        <div class="header-row">
            <span class="title">🔥 Dataset Filter</span>
            <span class="stats">
                <span class="stat stat-keep">✓ Kept: {kept}</span>
                <span class="stat stat-discard">✗ Discarded: {discarded}{auto_skipped_text}</span>
                <span class="stat stat-remaining">◎ {current + 1}/{total_samples}</span>
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Toggle for auto-skip (in sidebar or small)
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.auto_skip_invalid = st.checkbox(
            "Auto-skip invalid mermaid code", 
            value=st.session_state.auto_skip_invalid
        )
        st.caption(f"Found {len(invalid_codes)} invalid codes in dataset")
    
    # Progress bar (thin)
    st.progress(current / total_samples)
    
    current_caption = captions[current]
    current_code = codes[current]
    
    # Side by side: Instruction (25%) | Diagram (75%)
    col_instr, col_diagram = st.columns([1, 3])
    
    with col_instr:
        st.markdown(f"""
            <div class="instruction-label">📝 INSTRUCTION</div>
            <div class="instruction-box">{html.escape(current_caption)}</div>
        """, unsafe_allow_html=True)
        
        # Code expander under instruction
        with st.expander("View Code"):
            st.code(current_code, language="mermaid")
    
    with col_diagram:
        render_mermaid(current_code, height=500)
    
    # Buttons row
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c2:
        if st.button("👎 Discard", key="discard", use_container_width=True):
            st.session_state.discarded_count += 1
            st.session_state.current_index += 1
            st.rerun()
    with c3:
        if st.button("💾 Save & Exit", key="exit", use_container_width=True):
            # Save progress before exiting
            save_filtered_dataset(
                st.session_state.kept_indices, codes, captions,
                st.session_state.current_index, st.session_state.discarded_count
            )
            st.session_state.finished = True
            st.rerun()
    with c4:
        if st.button("👍 Keep", key="keep", use_container_width=True):
            st.session_state.kept_indices.append(current)
            st.session_state.current_index += 1
            st.rerun()
    
    st.markdown('<div class="keyboard-hint"><kbd>←</kbd>/<kbd>D</kbd> Discard | <kbd>→</kbd>/<kbd>K</kbd> Keep</div>', unsafe_allow_html=True)
    
    # Keyboard capture
    components.html("""
        <input type="text" id="kb" style="position:absolute;opacity:0;width:1px;height:1px;" autofocus />
        <script>
            const i = document.getElementById('kb');
            i.focus();
            setInterval(() => { if(document.activeElement !== i) i.focus(); }, 500);
            document.addEventListener('keydown', e => {
                const k = e.key.toLowerCase();
                if (k === 'arrowright' || k === 'k') {
                    e.preventDefault();
                    for (const b of window.parent.document.querySelectorAll('button'))
                        if (b.innerText.includes('Keep')) { b.click(); break; }
                } else if (k === 'arrowleft' || k === 'd') {
                    e.preventDefault();
                    for (const b of window.parent.document.querySelectorAll('button'))
                        if (b.innerText.includes('Discard')) { b.click(); break; }
                }
            });
        </script>
    """, height=0)


if __name__ == "__main__":
    main()

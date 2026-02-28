"""
ÉVALUATION COMPLÈTE - Pipeline Caption → Mermaid
==================================================

Ce script évalue le pipeline complet en 5 grandes dimensions :

  1. MÉTRIQUES DE GÉNÉRATION DE TEXTE  → BLEU, ROUGE, ChrF
  2. VALIDITÉ SYNTAXIQUE MERMAID       → Parsing du code généré
  3. SIMILARITÉ STRUCTURELLE           → Entités & relations (F1)
  4. ANALYSE DES EMBEDDINGS            → Cohérence des Graph Words
  5. ANALYSE D'ERREURS                 → Catégorisation des échecs

UTILISATION :
    python evaluate.py \
        --phase1_model  models/best_projection_module.pt \
        --phase2_model  models/phase2/best_complete_model.pt \
        --data          data-00000-of-00001.arrow \
        --output_dir    eval_results \
        --num_samples   200

DÉPENDANCES :
    pip install torch transformers sentence-transformers pyarrow
    pip install nltk rouge-score sacrebleu scikit-learn matplotlib seaborn tqdm
"""

import os
import json
import re
import argparse
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pyarrow as pa
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Métriques NLP
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

# Visualisation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Sklearn
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# Téléchargement NLTK silencieux
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# =============================================================================
# SECTION 1 : ARCHITECTURE (copie fidèle de phase1.py / phase2.py)
# =============================================================================

class TextToGraphProjection(nn.Module):
    def __init__(self, text_dim=384, graph_word_dim=256, num_graph_words=8):
        super().__init__()
        self.num_graph_words = num_graph_words
        self.projection = nn.Sequential(
            nn.Linear(text_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512),     nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, graph_word_dim * num_graph_words)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=graph_word_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(graph_word_dim)

    def forward(self, text_embeddings):
        batch_size = text_embeddings.shape[0]
        projected = self.projection(text_embeddings)
        graph_words = projected.view(batch_size, self.num_graph_words, -1)
        refined, _ = self.self_attention(graph_words, graph_words, graph_words)
        return self.layer_norm(graph_words + refined)


class MermaidDecoder(nn.Module):
    def __init__(self, graph_word_dim=256, num_graph_words=8, gpt2_model='gpt2'):
        super().__init__()
        self.num_graph_words = num_graph_words
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt2_hidden_size = self.gpt2.config.n_embd
        self.graph_to_gpt2 = nn.Linear(graph_word_dim, self.gpt2_hidden_size)

    def generate(self, graph_words, max_length=512, num_beams=5):
        batch_size = graph_words.shape[0]
        graph_embeddings = self.graph_to_gpt2(graph_words)
        attention_mask = torch.ones(batch_size, self.num_graph_words, device=graph_words.device)
        return self.gpt2.generate(
            inputs_embeds=graph_embeddings,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


class CaptionToMermaidModel(nn.Module):
    def __init__(self, projection_module, decoder):
        super().__init__()
        self.projection = projection_module
        self.decoder = decoder

    def generate(self, text_embeddings, max_length=512, num_beams=5):
        graph_words = self.projection(text_embeddings)
        return self.decoder.generate(graph_words, max_length=max_length, num_beams=num_beams)


# =============================================================================
# SECTION 2 : CHARGEMENT DES MODÈLES
# =============================================================================

def load_models(phase1_path: str, phase2_path: str, device: str):
    """Charge les modèles Phase 1 et Phase 2."""
    logger.info("Chargement du text encoder (all-MiniLM-L6-v2)...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info(f"Chargement du modèle Phase 1 depuis {phase1_path}...")
    ckpt1 = torch.load(phase1_path, map_location='cpu', weights_only=False)
    projection = TextToGraphProjection(text_dim=384, graph_word_dim=256, num_graph_words=8)
    projection.load_state_dict(ckpt1['model_state_dict'])
    logger.info(f"  ✓ Phase 1 chargée (epoch {ckpt1['epoch']}, val_loss={ckpt1['val_loss']:.4f})")

    logger.info(f"Chargement du modèle Phase 2 depuis {phase2_path}...")
    decoder = MermaidDecoder(graph_word_dim=256, num_graph_words=8, gpt2_model='gpt2')
    ckpt2 = torch.load(phase2_path, map_location='cpu', weights_only=False)
    model = CaptionToMermaidModel(projection, decoder)
    model.load_state_dict(ckpt2['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"  ✓ Phase 2 chargée (epoch {ckpt2['epoch']}, val_loss={ckpt2['val_loss']:.4f})")

    tokenizer = decoder.tokenizer
    return text_encoder, model, tokenizer


# =============================================================================
# SECTION 3 : GÉNÉRATION DES PRÉDICTIONS
# =============================================================================

def generate_predictions(
    model, text_encoder, tokenizer, captions: List[str],
    batch_size: int = 8, max_length: int = 512, num_beams: int = 5, device: str = 'cpu'
) -> List[str]:
    """Génère le code Mermaid pour une liste de captions."""
    predictions = []
    model.eval()

    for i in tqdm(range(0, len(captions), batch_size), desc="Génération"):
        batch_captions = captions[i:i + batch_size]
        embeddings = text_encoder.encode(batch_captions, convert_to_tensor=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(embeddings, max_length=max_length, num_beams=num_beams)

        for ids in output_ids:
            code = tokenizer.decode(ids, skip_special_tokens=True)
            # Nettoyage basique
            code = code.replace('```mermaid', '').replace('```', '').strip()
            predictions.append(code)

    return predictions


# =============================================================================
# SECTION 4 : MÉTRIQUES NLP
# =============================================================================

def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """BLEU-1 à BLEU-4 au niveau token."""
    smoother = SmoothingFunction().method1
    refs_tok  = [[ref.split()] for ref in references]
    hyps_tok  = [hyp.split() for hyp in hypotheses]

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f'bleu_{n}'] = corpus_bleu(refs_tok, hyps_tok, weights=weights,
                                           smoothing_function=smoother) * 100
    return scores


def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    agg = defaultdict(list)
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for k, v in scores.items():
            agg[k].append(v.fmeasure)
    return {k: np.mean(v) * 100 for k, v in agg.items()}


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """ChrF (Character n-gram F-score) — robuste aux erreurs syntaxiques."""
    if HAS_SACREBLEU:
        result = sacrebleu.corpus_chrf(hypotheses, [references])
        return result.score
    # Implémentation manuelle simplifiée
    def chrf_single(ref, hyp, n=6, beta=2):
        def ngrams(s, n):
            return Counter([s[i:i+n] for i in range(len(s)-n+1)])
        scores = []
        for k in range(1, n+1):
            rng = ngrams(ref, k)
            hng = ngrams(hyp, k)
            match = sum((rng & hng).values())
            p = match / max(sum(hng.values()), 1)
            r = match / max(sum(rng.values()), 1)
            if p + r == 0:
                scores.append(0)
            else:
                scores.append((1 + beta**2) * p * r / (beta**2 * p + r))
        return np.mean(scores) * 100
    return np.mean([chrf_single(r, h) for r, h in zip(references, hypotheses)])


# =============================================================================
# SECTION 5 : VALIDITÉ SYNTAXIQUE MERMAID
# =============================================================================

# Patterns des types de diagrammes Mermaid supportés
MERMAID_DIAGRAM_TYPES = [
    r'^flowchart\s+(TD|TB|BT|RL|LR)',
    r'^graph\s+(TD|TB|BT|RL|LR)',
    r'^sequenceDiagram',
    r'^classDiagram',
    r'^stateDiagram(-v2)?',
    r'^erDiagram',
    r'^gantt',
    r'^pie(\s+title)?',
    r'^gitGraph',
    r'^mindmap',
    r'^timeline',
    r'^C4Context',
    r'^flowchart\s*\n',
    r'^graph\s*\n',
]

# Patterns valides d'arêtes dans un flowchart
EDGE_PATTERNS = [
    r'\w[\w\s]*\s*--?>?\s*\w',           # A --> B
    r'\w[\w\s]*\s*---\s*\w',              # A --- B
    r'\w[\w\s]*\s*==>\s*\w',              # A ==> B
    r'\w[\w\s]*\s*-\.->\s*\w',            # A -.-> B
    r'\w[\w\s]*\s*--\|.*\|-->\s*\w',     # A --| text |--> B
]

# Patterns de nœuds valides
NODE_PATTERNS = [
    r'\w+\[.+\]',     # Rectangle
    r'\w+\(.+\)',     # Rond
    r'\w+\{.+\}',     # Losange
    r'\w+\[/.+/\]',   # Parallélogramme
    r'\w+\[\[.+\]\]', # Sous-routine
]


@dataclass
class SyntaxAnalysis:
    has_valid_header: bool = False
    diagram_type: str = "unknown"
    node_count: int = 0
    edge_count: int = 0
    has_edges: bool = False
    is_empty: bool = True
    syntax_score: float = 0.0          # Score global 0-100
    errors: List[str] = field(default_factory=list)


def analyze_mermaid_syntax(code: str) -> SyntaxAnalysis:
    """Analyse syntaxique détaillée d'un code Mermaid."""
    result = SyntaxAnalysis()

    if not code or len(code.strip()) < 5:
        result.is_empty = True
        result.errors.append("Code vide ou trop court")
        return result

    result.is_empty = False
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]

    if not lines:
        result.errors.append("Aucune ligne non vide")
        return result

    # Vérification du header
    first_line = lines[0].strip()
    for pattern in MERMAID_DIAGRAM_TYPES:
        if re.match(pattern, first_line, re.IGNORECASE):
            result.has_valid_header = True
            result.diagram_type = first_line.split()[0].lower()
            break

    if not result.has_valid_header:
        result.errors.append(f"Header invalide: '{first_line[:50]}'")

    # Comptage nœuds
    for line in lines[1:]:
        for pattern in NODE_PATTERNS:
            if re.search(pattern, line):
                result.node_count += 1
                break

    # Comptage arêtes
    for line in lines[1:]:
        for pattern in EDGE_PATTERNS:
            if re.search(pattern, line):
                result.edge_count += 1
                break

    result.has_edges = result.edge_count > 0

    # Score composite
    score = 0.0
    if result.has_valid_header:     score += 40
    if result.node_count >= 2:      score += 20
    if result.node_count >= 5:      score += 10
    if result.has_edges:            score += 20
    if result.edge_count >= 3:      score += 10
    result.syntax_score = min(score, 100.0)

    return result


def compute_syntax_metrics(predictions: List[str]) -> Dict:
    """Métriques syntaxiques agrégées sur toutes les prédictions."""
    analyses = [analyze_mermaid_syntax(p) for p in predictions]

    valid_headers   = sum(1 for a in analyses if a.has_valid_header)
    has_edges       = sum(1 for a in analyses if a.has_edges)
    non_empty       = sum(1 for a in analyses if not a.is_empty)
    avg_nodes       = np.mean([a.node_count for a in analyses])
    avg_edges       = np.mean([a.edge_count for a in analyses])
    avg_score       = np.mean([a.syntax_score for a in analyses])

    type_counts = Counter(a.diagram_type for a in analyses)

    return {
        'total_samples':         len(predictions),
        'non_empty_rate':        non_empty / len(predictions) * 100,
        'valid_header_rate':     valid_headers / len(predictions) * 100,
        'has_edges_rate':        has_edges / len(predictions) * 100,
        'avg_node_count':        avg_nodes,
        'avg_edge_count':        avg_edges,
        'avg_syntax_score':      avg_score,
        'diagram_type_dist':     dict(type_counts),
        'analyses':              analyses,
    }


# =============================================================================
# SECTION 6 : SIMILARITÉ STRUCTURELLE (F1 entités / relations)
# =============================================================================

def extract_mermaid_entities(code: str) -> Tuple[set, set]:
    """
    Extrait les identifiants de nœuds et les paires d'arêtes d'un code Mermaid.
    Retourne (nodes: set[str], edges: set[frozenset])
    """
    nodes = set()
    edges = set()

    # Nœuds avec label : A[Label], A(Label), A{Label}
    for match in re.finditer(r'(\w+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]', code):
        nodes.add(match.group(1).lower())

    # Arêtes : A --> B  /  A --- B  /  A ==> B  /  A -.-> B
    edge_re = re.compile(
        r'(\w+)\s*(?:-->|---?|==>|-.->|--\|[^|]*\|-->)\s*(\w+)',
        re.IGNORECASE
    )
    for match in edge_re.finditer(code):
        src, tgt = match.group(1).lower(), match.group(2).lower()
        nodes.add(src)
        nodes.add(tgt)
        edges.add(frozenset({src, tgt}))

    return nodes, edges


def structural_f1(ref_code: str, hyp_code: str) -> Dict[str, float]:
    """F1 pour les nœuds et les arêtes entre un code de référence et une prédiction."""
    ref_nodes, ref_edges = extract_mermaid_entities(ref_code)
    hyp_nodes, hyp_edges = extract_mermaid_entities(hyp_code)

    def f1(ref_set, hyp_set):
        if not ref_set and not hyp_set:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        if not ref_set or not hyp_set:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        tp = len(ref_set & hyp_set)
        p  = tp / len(hyp_set)
        r  = tp / len(ref_set)
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {'precision': p, 'recall': r, 'f1': f}

    return {
        'node_f1':  f1(ref_nodes, hyp_nodes)['f1'],
        'edge_f1':  f1(ref_edges, hyp_edges)['f1'],
        'node_precision': f1(ref_nodes, hyp_nodes)['precision'],
        'node_recall':    f1(ref_nodes, hyp_nodes)['recall'],
        'edge_precision': f1(ref_edges, hyp_edges)['precision'],
        'edge_recall':    f1(ref_edges, hyp_edges)['recall'],
    }


def compute_structural_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    """Métriques structurelles agrégées."""
    all_scores = [structural_f1(r, h) for r, h in zip(references, hypotheses)]
    keys = ['node_f1', 'edge_f1', 'node_precision', 'node_recall', 'edge_precision', 'edge_recall']
    return {k: np.mean([s[k] for s in all_scores]) * 100 for k in keys}


# =============================================================================
# SECTION 7 : COHÉRENCE DES GRAPH WORDS (Phase 1)
# =============================================================================

def evaluate_graph_words(
    model, text_encoder, captions: List[str], references: List[str],
    device: str = 'cpu', sample_size: int = 100
) -> Dict:
    """
    Évalue la qualité des Graph Words produits par le module de projection :
      - Cohérence intra-caption : même caption → Graph Words similaires
      - Séparabilité inter-caption : captions différentes → Graph Words différents
      - Corrélation caption / code : similarité cosinus (caption→GW) vs (code→GW)
    """
    model.eval()
    n = min(sample_size, len(captions))
    captions_sample   = captions[:n]
    references_sample = references[:n]

    # Obtenir les Graph Words
    all_gw = []
    with torch.no_grad():
        for caption in tqdm(captions_sample, desc="Graph Words encoding"):
            emb = text_encoder.encode([caption], convert_to_tensor=True).to(device)
            gw  = model.projection(emb)                     # (1, 8, 256)
            gw_flat = gw.squeeze(0).mean(dim=0).cpu().numpy()  # (256,)  — mean pooling
            all_gw.append(gw_flat)

    gw_matrix = np.array(all_gw)    # (n, 256)

    # Embedding des captions dans l'espace SBERT
    cap_embs  = text_encoder.encode(captions_sample)         # (n, 384)
    code_embs = text_encoder.encode(references_sample)       # (n, 384)

    # Similarité GW entre exemples similaires vs différents
    cos_sim_gw = cosine_similarity(gw_matrix)
    cap_sim    = cosine_similarity(cap_embs)

    # Corrélation rang de Spearman entre similarités
    upper_idx = np.triu_indices(n, k=1)
    gw_sims   = cos_sim_gw[upper_idx]
    cap_sims  = cap_sim[upper_idx]

    from scipy.stats import spearmanr
    corr, pval = spearmanr(gw_sims, cap_sims)

    # Diversité (moyenne de la distance intra-batch)
    diversity = 1.0 - np.mean(np.abs(cos_sim_gw[upper_idx]))

    return {
        'graph_words_caption_correlation': float(corr),
        'graph_words_caption_pvalue':      float(pval),
        'graph_words_diversity':           float(diversity),
        'avg_inter_similarity':            float(np.mean(gw_sims)),
    }


# =============================================================================
# SECTION 8 : ANALYSE D'ERREURS
# =============================================================================

ERROR_CATEGORIES = {
    'empty_output':        lambda pred, ref: len(pred.strip()) < 5,
    'wrong_diagram_type':  lambda pred, ref: (
        bool(re.match(r'^\w+', ref)) and
        bool(re.match(r'^\w+', pred)) and
        re.match(r'^\w+', ref).group(0).lower() != re.match(r'^\w+', pred).group(0).lower()
    ),
    'missing_header':      lambda pred, ref: not analyze_mermaid_syntax(pred).has_valid_header,
    'no_edges':            lambda pred, ref: not analyze_mermaid_syntax(pred).has_edges,
    'truncated':           lambda pred, ref: len(pred) < len(ref) * 0.3,
    'hallucinated':        lambda pred, ref: len(pred) > len(ref) * 3,
}


def categorize_errors(predictions: List[str], references: List[str]) -> Dict:
    """Catégorise les erreurs de chaque prédiction."""
    error_counts = defaultdict(int)
    error_examples = defaultdict(list)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        for cat, fn in ERROR_CATEGORIES.items():
            try:
                if fn(pred, ref):
                    error_counts[cat] += 1
                    if len(error_examples[cat]) < 3:
                        error_examples[cat].append({'idx': i, 'pred': pred[:200], 'ref': ref[:200]})
            except Exception:
                pass

    total = len(predictions)
    return {
        'error_rates': {k: v / total * 100 for k, v in error_counts.items()},
        'error_counts': dict(error_counts),
        'error_examples': dict(error_examples),
    }


# =============================================================================
# SECTION 9 : VISUALISATIONS
# =============================================================================

def plot_results(metrics: Dict, output_dir: str):
    """Génère et sauvegarde les graphiques d'évaluation."""
    if not HAS_PLOT:
        logger.warning("matplotlib/seaborn non disponibles, skip visualisations")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # ── Figure 1 : Métriques NLP ──────────────────────────────────────────────
    nlp = metrics.get('nlp_metrics', {})
    if nlp:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Métriques de génération de texte', fontsize=14, fontweight='bold')

        # BLEU
        bleu_keys = [k for k in nlp if k.startswith('bleu_')]
        if bleu_keys:
            vals = [nlp[k] for k in sorted(bleu_keys)]
            labs = [k.upper().replace('_', '-') for k in sorted(bleu_keys)]
            axes[0].bar(labs, vals, color='steelblue')
            axes[0].set_title('Scores BLEU')
            axes[0].set_ylabel('Score (%)')
            for bar, val in zip(axes[0].patches, vals):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                             f'{val:.1f}', ha='center', fontsize=9)

        # ROUGE
        rouge_keys = [k for k in nlp if k.startswith('rouge')]
        if rouge_keys:
            vals = [nlp[k] for k in sorted(rouge_keys)]
            labs = [k.upper() for k in sorted(rouge_keys)]
            axes[1].bar(labs, vals, color='darkorange')
            axes[1].set_title('Scores ROUGE')
            axes[1].set_ylabel('F1 Score (%)')
            for bar, val in zip(axes[1].patches, vals):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                             f'{val:.1f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nlp_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  ✓ nlp_metrics.png sauvegardé")

    # ── Figure 2 : Métriques syntaxiques ──────────────────────────────────────
    syn = metrics.get('syntax_metrics', {})
    if syn:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Analyse syntaxique Mermaid', fontsize=14, fontweight='bold')

        # Taux de validité
        rates = {
            'Non vide':       syn.get('non_empty_rate', 0),
            'Header valide':  syn.get('valid_header_rate', 0),
            'Arêtes présentes': syn.get('has_edges_rate', 0),
        }
        axes[0].barh(list(rates.keys()), list(rates.values()), color=['green', 'blue', 'purple'])
        axes[0].set_xlim(0, 105)
        axes[0].set_xlabel('Taux (%)')
        axes[0].set_title('Taux de validité')
        for i, v in enumerate(rates.values()):
            axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

        # Distribution des types de diagrammes
        type_dist = syn.get('diagram_type_dist', {})
        if type_dist:
            top_types = dict(Counter(type_dist).most_common(8))
            axes[1].pie(top_types.values(), labels=top_types.keys(), autopct='%1.1f%%',
                       startangle=90)
            axes[1].set_title('Distribution des types de diagrammes')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'syntax_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  ✓ syntax_metrics.png sauvegardé")

    # ── Figure 3 : F1 structurel ───────────────────────────────────────────────
    struct = metrics.get('structural_metrics', {})
    if struct:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_to_plot = {
            'Node\nPrecision': struct.get('node_precision', 0),
            'Node\nRecall':    struct.get('node_recall', 0),
            'Node\nF1':        struct.get('node_f1', 0),
            'Edge\nPrecision': struct.get('edge_precision', 0),
            'Edge\nRecall':    struct.get('edge_recall', 0),
            'Edge\nF1':        struct.get('edge_f1', 0),
        }
        colors = ['steelblue'] * 3 + ['darkorange'] * 3
        bars = ax.bar(list(metrics_to_plot.keys()), list(metrics_to_plot.values()), color=colors)
        ax.set_ylabel('Score (%)')
        ax.set_title('Similarité structurelle (Entités & Relations)')
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, metrics_to_plot.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'structural_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  ✓ structural_metrics.png sauvegardé")

    # ── Figure 4 : Analyse d'erreurs ───────────────────────────────────────────
    err = metrics.get('error_analysis', {})
    if err and err.get('error_rates'):
        fig, ax = plt.subplots(figsize=(8, 5))
        rates = err['error_rates']
        keys  = list(rates.keys())
        vals  = [rates[k] for k in keys]
        labels = [k.replace('_', '\n') for k in keys]
        bars = ax.bar(labels, vals, color='crimson', alpha=0.8)
        ax.set_ylabel('Taux d\'erreur (%)')
        ax.set_title('Analyse d\'erreurs par catégorie')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  ✓ error_analysis.png sauvegardé")


# =============================================================================
# SECTION 10 : RAPPORT FINAL
# =============================================================================

def print_report(metrics: Dict):
    """Affiche un rapport formaté dans le terminal."""
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  RAPPORT D'EVALUATION -- Pipeline Caption->Mermaid")
    print(SEP)

    # NLP
    nlp = metrics.get('nlp_metrics', {})
    if nlp:
        print("\n MÉTRIQUES NLP")
        print("-" * 40)
        for k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
            if k in nlp:
                print(f"  {k.upper().replace('_','-'):12s}: {nlp[k]:.2f}%")
        for k in ['rouge1', 'rouge2', 'rougeL']:
            if k in nlp:
                print(f"  {k.upper():12s}: {nlp[k]:.2f}%")
        if 'chrf' in nlp:
            print(f"  {'ChrF':12s}: {nlp['chrf']:.2f}")

    # Syntaxe
    syn = metrics.get('syntax_metrics', {})
    if syn:
        print("\n VALIDITÉ SYNTAXIQUE MERMAID")
        print("-" * 40)
        print(f"  Non vide         : {syn.get('non_empty_rate', 0):.1f}%")
        print(f"  Header valide    : {syn.get('valid_header_rate', 0):.1f}%")
        print(f"  Arêtes présentes : {syn.get('has_edges_rate', 0):.1f}%")
        print(f"  Score moyen      : {syn.get('avg_syntax_score', 0):.1f}/100")
        print(f"  Noeuds (moy.)    : {syn.get('avg_node_count', 0):.1f}")
        print(f"  Arêtes (moy.)    : {syn.get('avg_edge_count', 0):.1f}")
        dist = syn.get('diagram_type_dist', {})
        if dist:
            top3 = Counter(dist).most_common(3)
            print(f"  Types fréquents  : {', '.join(f'{t}({c})' for t, c in top3)}")

    # Structurel
    struct = metrics.get('structural_metrics', {})
    if struct:
        print("\n SIMILARITÉ STRUCTURELLE (entités & relations)")
        print("-" * 40)
        print(f"  Node F1          : {struct.get('node_f1', 0):.1f}%")
        print(f"  Node Precision   : {struct.get('node_precision', 0):.1f}%")
        print(f"  Node Recall      : {struct.get('node_recall', 0):.1f}%")
        print(f"  Edge F1          : {struct.get('edge_f1', 0):.1f}%")
        print(f"  Edge Precision   : {struct.get('edge_precision', 0):.1f}%")
        print(f"  Edge Recall      : {struct.get('edge_recall', 0):.1f}%")

    # Graph Words
    gw = metrics.get('graph_words_metrics', {})
    if gw:
        print("\n ANALYSE DES GRAPH WORDS (Phase 1)")
        print("-" * 40)
        corr = gw.get('graph_words_caption_correlation', 0)
        pval = gw.get('graph_words_caption_pvalue', 1)
        div  = gw.get('graph_words_diversity', 0)
        print(f"  Corrélation caption/GW : {corr:.3f}  (p={pval:.4f})")
        print(f"  Diversité GW           : {div:.3f}")
        print(f"  Sim. inter-moyenne     : {gw.get('avg_inter_similarity', 0):.3f}")
        if corr > 0.5:
            print("  ✓ Bonne corrélation — le module de projection est cohérent")
        elif corr > 0.3:
            print("  ⚠ Corrélation modérée — des améliorations sont possibles")
        else:
            print("  ✗ Corrélation faible — le module de projection peine à capturer le sens")

    # Erreurs
    err = metrics.get('error_analysis', {})
    if err and err.get('error_rates'):
        print("\n  ANALYSE D'ERREURS")
        print("-" * 40)
        for cat, rate in sorted(err['error_rates'].items(), key=lambda x: -x[1]):
            print(f"  {cat:25s}: {rate:.1f}%")

    print(f"\n{SEP}\n")


def save_report(metrics: Dict, output_dir: str):
    """Sauvegarde le rapport complet en JSON (sans les objets non-sérialisables)."""

    def sanitize(obj):
        if isinstance(obj, (np.integer, np.int64)):    return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):                return obj.tolist()
        if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items() if k != 'analyses'}
        if isinstance(obj, list):  return [sanitize(v) for v in obj]
        return obj

    path = os.path.join(output_dir, 'evaluation_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitize(metrics), f, indent=2, ensure_ascii=False)
    logger.info(f"Rapport JSON sauvegardé → {path}")


def save_qualitative_samples(
    captions: List[str], references: List[str], predictions: List[str],
    output_dir: str, n: int = 20
):
    """Sauvegarde n exemples qualitatifs pour inspection manuelle."""
    samples = []
    for i in range(min(n, len(captions))):
        ref_analysis  = analyze_mermaid_syntax(references[i])
        hyp_analysis  = analyze_mermaid_syntax(predictions[i])
        struct        = structural_f1(references[i], predictions[i])
        samples.append({
            'index':            i,
            'caption':          captions[i],
            'reference':        references[i],
            'prediction':       predictions[i],
            'ref_syntax_score': ref_analysis.syntax_score,
            'pred_syntax_score': hyp_analysis.syntax_score,
            'node_f1':          struct['node_f1'],
            'edge_f1':          struct['edge_f1'],
        })

    path = os.path.join(output_dir, 'qualitative_samples.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Exemples qualitatifs sauvegardés → {path}")


# =============================================================================
# SECTION 11 : POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Évaluation complète Phase 1 + Phase 2')
    parser.add_argument('--phase1_model',  required=True,  help='Chemin vers best_projection_module.pt')
    parser.add_argument('--phase2_model',  required=True,  help='Chemin vers best_complete_model.pt')
    parser.add_argument('--data',          required=True,  help='Chemin vers le fichier .arrow')
    parser.add_argument('--output_dir',    default='eval_results', help='Dossier de sortie')
    parser.add_argument('--num_samples',   type=int, default=200, help='Nombre d\'exemples à évaluer')
    parser.add_argument('--batch_size',    type=int, default=4,   help='Taille de batch pour la génération')
    parser.add_argument('--max_length',    type=int, default=512, help='Longueur max du code généré')
    parser.add_argument('--num_beams',     type=int, default=5,   help='Nombre de beams (1=greedy, rapide sur CPU)')
    parser.add_argument('--seed',          type=int, default=42,  help='Seed pour reproductibilité')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device : {device}")

    # ── Chargement du dataset ─────────────────────────────────────────────────
    logger.info(f"Chargement du dataset depuis {args.data}...")
    with pa.OSFile(args.data, 'rb') as src:
        table = pa.ipc.open_stream(src).read_all()
    df = table.to_pandas()
    logger.info(f"  ✓ {len(df)} exemples chargés")

    # Utilisation du split test (10% finaux, même seed que phase1/2)
    indices   = np.random.permutation(len(df))
    test_idx  = indices[int(len(df) * 0.9):]
    n         = min(args.num_samples, len(test_idx))
    test_idx  = test_idx[:n]

    captions   = df.iloc[test_idx]['caption'].tolist()
    references = df.iloc[test_idx]['code'].tolist()
    # Nettoyage références
    references = [r.replace('```mermaid\n', '').replace('\n```', '').strip() for r in references]

    logger.info(f"  ✓ {n} exemples de test sélectionnés")

    # ── Chargement des modèles ────────────────────────────────────────────────
    text_encoder, model, tokenizer = load_models(args.phase1_model, args.phase2_model, device)

    # ── Génération ───────────────────────────────────────────────────────────
    logger.info("Génération des prédictions...")
    predictions = generate_predictions(
        model, text_encoder, tokenizer, captions,
        batch_size=args.batch_size, max_length=args.max_length,
        num_beams=args.num_beams, device=device
    )

    # ── Calcul des métriques ──────────────────────────────────────────────────
    all_metrics = {}

    logger.info("Calcul des métriques NLP (BLEU, ROUGE, ChrF)...")
    nlp = {}
    nlp.update(compute_bleu(references, predictions))
    nlp.update(compute_rouge(references, predictions))
    nlp['chrf'] = compute_chrf(references, predictions)
    all_metrics['nlp_metrics'] = nlp

    logger.info("Analyse syntaxique Mermaid...")
    all_metrics['syntax_metrics'] = compute_syntax_metrics(predictions)
    # Enlever les objets non-sérialisables du dict principal
    all_metrics['syntax_metrics'].pop('analyses', None)

    logger.info("Calcul de la similarité structurelle (F1)...")
    all_metrics['structural_metrics'] = compute_structural_metrics(references, predictions)

    logger.info("Évaluation de la cohérence des Graph Words...")
    try:
        all_metrics['graph_words_metrics'] = evaluate_graph_words(
            model, text_encoder, captions, references, device=device, sample_size=min(100, n)
        )
    except Exception as e:
        logger.warning(f"Impossible d'évaluer les Graph Words : {e}")

    logger.info("Analyse d'erreurs...")
    all_metrics['error_analysis'] = categorize_errors(predictions, references)

    # ── Affichage & sauvegarde ────────────────────────────────────────────────
    print_report(all_metrics)
    save_report(all_metrics, args.output_dir)
    save_qualitative_samples(captions, references, predictions, args.output_dir)

    logger.info("Génération des visualisations...")
    plot_results(all_metrics, args.output_dir)

    logger.info(f"\n Évaluation terminée ! Résultats dans : {args.output_dir}/")
    logger.info("  Fichiers générés :")
    for fname in ['evaluation_report.json', 'qualitative_samples.json',
                  'nlp_metrics.png', 'syntax_metrics.png',
                  'structural_metrics.png', 'error_analysis.png']:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            logger.info(f"    ✓ {fname}")


if __name__ == "__main__":
    main()
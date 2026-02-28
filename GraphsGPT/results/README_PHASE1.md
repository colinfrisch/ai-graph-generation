# Phase 1: Entraînement du Projection Module

## Objectif
Entraîner un module de projection qui transforme les descriptions textuelles (captions) en Graph Words, permettant ensuite de générer du code Mermaid.

##  Prérequis

### Installation des dépendances
```bash
pip install torch transformers sentence-transformers datasets pyarrow scikit-learn tqdm tensorboard
```

### Vérification de votre environnement
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

##  Exécution Rapide (Quick Start)

### Option 1: Script complet automatique
```bash
python GraphsGPT/phase1_complete_training.py
```

Ce script exécute **toutes** les étapes automatiquement:
1. ✓ Charge le dataset Mermaid
2. ✓ Initialise les encodeurs (texte + graphe)
3. ✓ Crée le projection module
4. ✓ Entraîne pendant 100 epochs avec early stopping
5. ✓ Sauvegarde le meilleur modèle


## Architecture

```
Caption (texte)
    ↓
Text Encoder (Sentence-Transformers)
    ↓ 384D embeddings
Projection Module (VOTRE CONTRIBUTION)
    ↓ 8 x 256D Graph Words
Graph Decoder (GraphsGPT - Phase 2)
    ↓
Code Mermaid
```

### Composants Clés

1. **Text Encoder** (`all-MiniLM-L6-v2`)
   - Pré-entraîné sur 1B+ phrases
   - Sortie: 384D embeddings
   - Rapide et efficace

2. **Projection Module** (Ce que vous entraînez)
   ```python
   Input:  text_embeddings (batch, 384)
   Output: graph_words (batch, 8, 256)
   ```
   - 3 couches MLP + LayerNorm
   - Self-attention pour raffinement
   - Connexions résiduelles

3. **Graph Encoder** (Génération des targets)
   - Encode le code Mermaid en Graph Words
   - Architecture Transformer
   - Cross-attention pour extraction

##  Structure des Fichiers

```
graphsgpt_project/
├── data/                          # Datasets (copier votre .arrow ici)
├── models/                        # Modèles sauvegardés
│   ├── best_projection_module.pt  # Meilleur modèle
│   └── checkpoint_epoch_*.pt      # Checkpoints périodiques
├── outputs/                       # Logs et résultats
│   └── training_history.json      # Historique d'entraînement
├── scripts/
│   ├── phase1_complete_training.py    # Script complet
│   └── utils.py                       # Fonctions utilitaires
└── notebooks/
    └── phase1_training.ipynb          # Notebook interactif
```

##  Configuration

Modifiez la classe `Config` dans le script pour ajuster:

```python
class Config:
    # Text Encoder
    TEXT_ENCODER = 'all-MiniLM-L6-v2'  # Ou 'bert-base-uncased'
    TEXT_DIM = 384  # 384 pour MiniLM, 768 pour BERT
    
    # Graph Words
    NUM_GRAPH_WORDS = 8    # Nombre de graph words (4, 8, ou 16)
    GRAPH_WORD_DIM = 256   # Dimension de chaque graph word
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
```


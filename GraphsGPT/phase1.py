"""
PHASE 1: PROJECTION MODULE TRAINING - COMPLETE GUIDE
=====================================================

This script provides the complete implementation for Phase 1.
Follow each section in order.

REQUIREMENTS:
- torch >= 2.0.0
- transformers >= 4.35.0
- sentence-transformers >= 2.2.2
- datasets >= 2.14.0
- pyarrow >= 13.0.0

Install with:
pip install torch transformers sentence-transformers datasets pyarrow scikit-learn tqdm

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel  # Not used with MiniLM
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import numpy as np
from tqdm import tqdm
import os
import json

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    """Configuration for Phase 1 training"""

    # Paths (relative to script directory)
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(_SCRIPT_DIR, 'data-00000-of-00001.arrow')
    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
    MODEL_DIR = os.path.join(_SCRIPT_DIR, 'models')
    
    # Text Encoder Settings
    TEXT_ENCODER = 'all-MiniLM-L6-v2'  # Fast and good quality (384D)
    # Alternative: 'bert-base-uncased' (768D, slower but better)
    TEXT_DIM = 384  # Output dimension of text encoder
    
    # Graph Words Settings
    NUM_GRAPH_WORDS = 8  # Number of graph words to generate
    GRAPH_WORD_DIM = 256  # Dimension of each graph word
    
    # Training Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # Data Split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5


config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Device: {config.DEVICE}")
print(f"Text Encoder: {config.TEXT_ENCODER}")
print(f"Text Dim: {config.TEXT_DIM}")
print(f"Graph Words: {config.NUM_GRAPH_WORDS} x {config.GRAPH_WORD_DIM}D")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Learning Rate: {config.LEARNING_RATE}")
print(f"Epochs: {config.NUM_EPOCHS}")


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_mermaid_dataset(file_path):
    """Load the Mermaid dataset from Arrow format"""
    print("\n" + "=" * 80)
    print("LOADING MERMAID DATASET")
    print("=" * 80)
    
    with pa.OSFile(file_path, 'rb') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
    
    df = table.to_pandas()
    
    print(f"✓ Loaded {len(df)} examples")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Basic statistics
    caption_lengths = df['caption'].str.len()
    code_lengths = df['code'].str.len()
    
    print(f"\nCaption lengths: min={caption_lengths.min()}, "
          f"max={caption_lengths.max()}, mean={caption_lengths.mean():.1f}")
    print(f"Code lengths: min={code_lengths.min()}, "
          f"max={code_lengths.max()}, mean={code_lengths.mean():.1f}")
    
    return df


def split_dataset(df, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    """Split dataset into train/val/test"""
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    
    train_size = int(len(df) * train_split)
    val_size = int(len(df) * val_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return {
        'train': df.iloc[train_indices].reset_index(drop=True),
        'val': df.iloc[val_indices].reset_index(drop=True),
        'test': df.iloc[test_indices].reset_index(drop=True)
    }


# =============================================================================
# SECTION 3: GRAPH WORDS ENCODER (Target Generator)
# =============================================================================

class GraphWordsEncoder(nn.Module):
    """
    Encoder that generates Graph Words from Mermaid code.
    This is used to create training targets.
    
    In the real GraphsGPT, this would be the pre-trained encoder.
    Here we implement a custom version that learns to encode graphs.
    """
    def __init__(self, input_dim=768, graph_word_dim=256, num_graph_words=8):
        super().__init__()
        self.num_graph_words = num_graph_words
        self.graph_word_dim = graph_word_dim
        
        # Learnable graph word queries
        self.graph_word_queries = nn.Parameter(
            torch.randn(num_graph_words, graph_word_dim)
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, graph_word_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=graph_word_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Cross-attention to extract graph words
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=graph_word_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(graph_word_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - Encoded graph features
        Returns:
            graph_words: (batch_size, num_graph_words, graph_word_dim)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_projection(x)
        
        # Encode with transformer
        x = self.transformer_encoder(x)
        
        # Expand queries for batch
        queries = self.graph_word_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Extract graph words via cross-attention
        graph_words, _ = self.cross_attention(query=queries, key=x, value=x)
        
        # Layer norm
        graph_words = self.layer_norm(graph_words)
        
        return graph_words


# =============================================================================
# SECTION 4: TEXT-TO-GRAPH PROJECTION MODULE (What You'll Train)
# =============================================================================

class TextToGraphProjection(nn.Module):
    """
    YOUR CUSTOM MODULE: Projects text embeddings to graph words.
    This is the main contribution of your work.
    """
    def __init__(self, text_dim=384, graph_word_dim=256, num_graph_words=8):
        super().__init__()
        self.num_graph_words = num_graph_words
        
        # Multi-layer projection with residual connections
        self.projection = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, graph_word_dim * num_graph_words)
        )
        
        # Additional refinement with self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=graph_word_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(graph_word_dim)
        
    def forward(self, text_embeddings):
        """
        Args:
            text_embeddings: (batch_size, text_dim)
        Returns:
            graph_words: (batch_size, num_graph_words, graph_word_dim)
        """
        batch_size = text_embeddings.shape[0]
        
        # Project to graph words
        projected = self.projection(text_embeddings)
        graph_words = projected.view(batch_size, self.num_graph_words, -1)
        
        # Refine with self-attention
        refined, _ = self.self_attention(
            query=graph_words,
            key=graph_words,
            value=graph_words
        )
        
        # Residual connection
        graph_words = graph_words + refined
        graph_words = self.layer_norm(graph_words)
        
        return graph_words


# =============================================================================
# SECTION 5: DATASET CLASS
# =============================================================================

class MermaidDataset(Dataset):
    """Dataset for training the projection module"""
    
    def __init__(self, df, text_encoder, graph_encoder=None, mode='train'):
        """
        Args:
            df: DataFrame with 'caption' and 'code' columns
            text_encoder: Pre-trained text encoder
            graph_encoder: Graph encoder for generating targets (None for inference)
            mode: 'train', 'val', or 'test'
        """
        self.df = df
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.mode = mode
        
        # Pre-compute text embeddings
        print(f"Pre-computing text embeddings for {mode} set...")
        self.text_embeddings = []
        for caption in tqdm(df['caption'], desc=f"Encoding {mode} captions"):
            embedding = text_encoder.encode(caption, convert_to_tensor=True)
            self.text_embeddings.append(embedding.cpu())
        
        # Pre-compute graph words (targets) if graph encoder provided
        if graph_encoder is not None and mode in ('train', 'val'):
            print(f"Pre-computing graph words targets for {mode} set...")
            self.graph_words_targets = []
            graph_encoder.eval()
            
            with torch.no_grad():
                for code in tqdm(df['code'], desc=f"Encoding {mode} graphs"):
                    # For now, use text encoder on code as a proxy
                    # In real GraphsGPT, this would use the graph encoder
                    code_embedding = text_encoder.encode(code, convert_to_tensor=True)
                    
                    # Add sequence dimension and pass through graph encoder
                    code_embedding = code_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
                    
                    # Move to same device as graph encoder
                    device = next(graph_encoder.parameters()).device
                    code_embedding = code_embedding.to(device)
                    
                    graph_words = graph_encoder(code_embedding)
                    self.graph_words_targets.append(graph_words.squeeze(0).cpu())
        else:
            self.graph_words_targets = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            'caption': self.df.iloc[idx]['caption'],
            'code': self.df.iloc[idx]['code'],
            'text_embedding': self.text_embeddings[idx]
        }
        
        if self.graph_words_targets is not None:
            item['graph_words_target'] = self.graph_words_targets[idx]
        
        return item


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

def train_projection_module(
    projection_module,
    graph_encoder,
    train_loader,
    val_loader,
    config
):
    """Train the text-to-graph projection module"""
    
    projection_module = projection_module.to(config.DEVICE)
    graph_encoder = graph_encoder.to(config.DEVICE)
    graph_encoder.eval()  # Freeze graph encoder
    
    # Optimizer
    optimizer = optim.AdamW(
        projection_module.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        projection_module.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in pbar:
            text_embeddings = batch['text_embedding'].to(config.DEVICE)
            graph_words_target = batch['graph_words_target'].to(config.DEVICE)
            
            # Forward pass
            predicted_graph_words = projection_module(text_embeddings)
            
            # Compute loss
            loss = criterion(predicted_graph_words, graph_words_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection_module.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        projection_module.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_embeddings = batch['text_embedding'].to(config.DEVICE)
                graph_words_target = batch['graph_words_target'].to(config.DEVICE)
                
                predicted_graph_words = projection_module(text_embeddings)
                loss = criterion(predicted_graph_words, graph_words_target)
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': projection_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            }, os.path.join(config.MODEL_DIR, 'best_projection_module.pt'))
            
            print(f"  ✓ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': projection_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save training history
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("PHASE 1: PROJECTION MODULE TRAINING")
    print("=" * 80)
    
    # Step 1: Load dataset
    df = load_mermaid_dataset(config.DATA_PATH)
    
    # Step 2: Split dataset
    splits = split_dataset(
        df, 
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val: {len(splits['val'])} examples")
    print(f"  Test: {len(splits['test'])} examples")
    
    # Step 3: Initialize text encoder
    print("\nLoading text encoder...")
    text_encoder = SentenceTransformer(config.TEXT_ENCODER)
    print(f"✓ Text encoder loaded: {config.TEXT_ENCODER}")
    
    # Step 4: Initialize graph encoder (for generating targets)
    print("\nInitializing graph encoder...")
    graph_encoder = GraphWordsEncoder(
        input_dim=config.TEXT_DIM,
        graph_word_dim=config.GRAPH_WORD_DIM,
        num_graph_words=config.NUM_GRAPH_WORDS
    )
    print(f"✓ Graph encoder initialized")
    
    # Step 5: Create datasets
    train_dataset = MermaidDataset(
        splits['train'], text_encoder, graph_encoder, mode='train'
    )
    val_dataset = MermaidDataset(
        splits['val'], text_encoder, graph_encoder, mode='val'
    )
    
    # Step 6: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Step 7: Initialize projection module
    projection_module = TextToGraphProjection(
        text_dim=config.TEXT_DIM,
        graph_word_dim=config.GRAPH_WORD_DIM,
        num_graph_words=config.NUM_GRAPH_WORDS
    )
    
    print(f"\n✓ Projection module initialized")
    print(f"  Parameters: {sum(p.numel() for p in projection_module.parameters()):,}")
    
    # Step 8: Train!
    history = train_projection_module(
        projection_module,
        graph_encoder,
        train_loader,
        val_loader,
        config
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {config.MODEL_DIR}/best_projection_module.pt")
    print(f"Training history saved to: {config.OUTPUT_DIR}/training_history.json")


if __name__ == "__main__":
    main()

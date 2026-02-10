"""
PHASE 2: END-TO-END TRAINING - Caption to Mermaid Code
========================================================

This script implements the complete pipeline:
Caption → Projection Module → Graph Words → GPT-2 Decoder → Mermaid Code

REQUIREMENTS:
- Phase 1 must be completed successfully
- best_projection_module.pt must exist in models/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
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
    """Configuration for Phase 2 training"""

    # Paths (relative to script directory)
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(_SCRIPT_DIR, 'data-00000-of-00001.arrow')
    PHASE1_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'models', 'best_projection_module.pt')
    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs', 'phase2')
    MODEL_DIR = os.path.join(_SCRIPT_DIR, 'models', 'phase2')
    
    # Text Encoder (from Phase 1)
    TEXT_ENCODER = 'all-MiniLM-L6-v2'
    TEXT_DIM = 384
    
    # Graph Words (from Phase 1)
    NUM_GRAPH_WORDS = 8
    GRAPH_WORD_DIM = 256
    
    # GPT-2 Decoder Settings
    GPT2_MODEL = 'gpt2'  # or 'gpt2-medium' for better quality
    MAX_LENGTH = 512  # Maximum sequence length for Mermaid code
    
    # Training Settings
    BATCH_SIZE = 8  # Smaller because of GPT-2
    LEARNING_RATE = 5e-5  # Lower for fine-tuning
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    
    # Progressive Unfreezing
    FREEZE_PROJECTION_EPOCHS = 0  # 0 = train everything from start
    # Set to 5-10 if you want to train decoder first, then unfreeze projection
    
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
print("PHASE 2: END-TO-END TRAINING")
print("=" * 80)
print(f"Device: {config.DEVICE}")
print(f"GPT-2 Model: {config.GPT2_MODEL}")
print(f"Max Length: {config.MAX_LENGTH}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Learning Rate: {config.LEARNING_RATE}")


# =============================================================================
# SECTION 2: LOAD PHASE 1 COMPONENTS
# =============================================================================

class TextToGraphProjection(nn.Module):
    """Projection Module from Phase 1 (same as before)"""
    def __init__(self, text_dim=384, graph_word_dim=256, num_graph_words=8):
        super().__init__()
        self.num_graph_words = num_graph_words
        
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
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=graph_word_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(graph_word_dim)
        
    def forward(self, text_embeddings):
        batch_size = text_embeddings.shape[0]
        projected = self.projection(text_embeddings)
        graph_words = projected.view(batch_size, self.num_graph_words, -1)
        refined, _ = self.self_attention(graph_words, graph_words, graph_words)
        graph_words = graph_words + refined
        graph_words = self.layer_norm(graph_words)
        return graph_words


def load_phase1_model(checkpoint_path, config):
    """Load the trained Projection Module from Phase 1"""
    print("\n" + "=" * 80)
    print("LOADING PHASE 1 MODEL")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {checkpoint_path}\n"
            "Please run phase1_complete_training.py first!"
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    projection_module = TextToGraphProjection(
        text_dim=config.TEXT_DIM,
        graph_word_dim=config.GRAPH_WORD_DIM,
        num_graph_words=config.NUM_GRAPH_WORDS
    )
    
    projection_module.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded Phase 1 model from epoch {checkpoint['epoch']}")
    print(f"✓ Phase 1 Val Loss: {checkpoint['val_loss']:.6f}")
    
    return projection_module


# =============================================================================
# SECTION 3: GPT-2 DECODER WITH GRAPH WORDS CONDITIONING
# =============================================================================

class MermaidDecoder(nn.Module):
    """GPT-2 based decoder conditioned on Graph Words"""
    
    def __init__(self, graph_word_dim=256, num_graph_words=8, gpt2_model='gpt2'):
        super().__init__()
        self.num_graph_words = num_graph_words
        
        # Load pre-trained GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt2_hidden_size = self.gpt2.config.n_embd
        
        # Project Graph Words to GPT-2 embedding space
        self.graph_to_gpt2 = nn.Linear(graph_word_dim, self.gpt2_hidden_size)
        
        print(f"\n✓ GPT-2 Decoder initialized")
        print(f"  - Model: {gpt2_model}")
        print(f"  - Hidden size: {self.gpt2_hidden_size}")
        print(f"  - Vocab size: {len(self.tokenizer)}")
        
    def forward(self, graph_words, input_ids=None, attention_mask=None, labels=None):
        """
        Args:
            graph_words: (batch_size, num_graph_words, graph_word_dim)
            input_ids: (batch_size, seq_len) - Mermaid code tokens
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) - For training
        """
        batch_size = graph_words.shape[0]
        
        # Project graph words to GPT-2 space
        graph_embeddings = self.graph_to_gpt2(graph_words)  # (batch, num_graph_words, hidden)
        
        if input_ids is not None:
            # Get GPT-2 embeddings for input tokens
            inputs_embeds = self.gpt2.transformer.wte(input_ids)  # (batch, seq_len, hidden)
            
            # Concatenate graph embeddings at the beginning
            inputs_embeds = torch.cat([graph_embeddings, inputs_embeds], dim=1)
            
            # Extend attention mask for graph words
            if attention_mask is not None:
                graph_attention = torch.ones(
                    batch_size, self.num_graph_words,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([graph_attention, attention_mask], dim=1)
            
            # CRITICAL FIX: Adjust labels to match the concatenated sequence length
            # Add padding for graph words at the beginning (set to -100 to ignore in loss)
            if labels is not None:
                graph_labels_padding = torch.full(
                    (batch_size, self.num_graph_words),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([graph_labels_padding, labels], dim=1)
        else:
            # Generation mode: only graph embeddings
            inputs_embeds = graph_embeddings
            attention_mask = torch.ones(
                batch_size, self.num_graph_words,
                device=graph_words.device
            )
        
        # Forward through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None
        )
        
        return outputs
    
    def generate(self, graph_words, max_length=512, num_beams=5):
        """Generate Mermaid code from graph words"""
        batch_size = graph_words.shape[0]
        
        # Project graph words
        graph_embeddings = self.graph_to_gpt2(graph_words)
        
        # Start with graph embeddings
        attention_mask = torch.ones(
            batch_size, self.num_graph_words,
            device=graph_words.device
        )
        
        # Generate
        outputs = self.gpt2.generate(
            inputs_embeds=graph_embeddings,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs


# =============================================================================
# SECTION 4: COMPLETE END-TO-END MODEL
# =============================================================================

class CaptionToMermaidModel(nn.Module):
    """Complete model: Caption → Graph Words → Mermaid Code"""
    
    def __init__(self, projection_module, decoder):
        super().__init__()
        self.projection = projection_module
        self.decoder = decoder
        
    def forward(self, text_embeddings, input_ids=None, attention_mask=None, labels=None):
        # Get graph words from projection module
        graph_words = self.projection(text_embeddings)
        
        # Decode to Mermaid code
        outputs = self.decoder(
            graph_words,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, text_embeddings, max_length=512, num_beams=5):
        graph_words = self.projection(text_embeddings)
        return self.decoder.generate(graph_words, max_length=max_length, num_beams=num_beams)


# =============================================================================
# SECTION 5: DATASET
# =============================================================================

class MermaidDatasetPhase2(Dataset):
    """Dataset for Phase 2 end-to-end training"""
    
    def __init__(self, df, text_encoder, tokenizer, max_length=512):
        self.df = df
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-compute text embeddings
        print("Pre-computing text embeddings...")
        self.text_embeddings = []
        for caption in tqdm(df['caption']):
            embedding = text_encoder.encode(caption, convert_to_tensor=True)
            self.text_embeddings.append(embedding.cpu())
        
        # Tokenize Mermaid codes
        print("Tokenizing Mermaid codes...")
        self.tokenized_codes = []
        for code in tqdm(df['code']):
            # Remove markdown code blocks
            code = code.replace('```mermaid\n', '').replace('\n```', '').strip()
            
            tokens = tokenizer(
                code,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.tokenized_codes.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0)
            })
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'text_embedding': self.text_embeddings[idx],
            'input_ids': self.tokenized_codes[idx]['input_ids'],
            'attention_mask': self.tokenized_codes[idx]['attention_mask'],
            'labels': self.tokenized_codes[idx]['input_ids'].clone()  # For loss computation
        }


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

def train_phase2(model, train_loader, val_loader, config):
    """Train the complete end-to-end model"""
    
    model = model.to(config.DEVICE)
    
    # Optimizer - different learning rates for different parts
    projection_params = list(model.projection.parameters())
    decoder_params = list(model.decoder.parameters())
    
    optimizer = optim.AdamW([
        {'params': projection_params, 'lr': config.LEARNING_RATE},
        {'params': decoder_params, 'lr': config.LEARNING_RATE}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("STARTING PHASE 2 TRAINING")
    print("=" * 80)
    
    for epoch in range(config.NUM_EPOCHS):
        # Optionally freeze projection module for first few epochs
        if epoch < config.FREEZE_PROJECTION_EPOCHS:
            for param in model.projection.parameters():
                param.requires_grad = False
            print(f"[Epoch {epoch+1}] Projection module FROZEN")
        else:
            for param in model.projection.parameters():
                param.requires_grad = True
        
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in pbar:
            text_embeddings = batch['text_embedding'].to(config.DEVICE)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            # Forward pass
            outputs = model(
                text_embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_embeddings = batch['text_embedding'].to(config.DEVICE)
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                
                outputs = model(
                    text_embeddings,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_losses.append(outputs.loss.item())
        
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            }, os.path.join(config.MODEL_DIR, 'best_complete_model.pt'))
            
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save training history
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


# =============================================================================
# SECTION 7: GENERATION AND EVALUATION
# =============================================================================

def generate_samples(model, test_dataset, tokenizer, num_samples=5):
    """Generate Mermaid code for sample captions"""
    
    model.eval()
    
    print("\n" + "=" * 80)
    print("GENERATING SAMPLES")
    print("=" * 80)
    
    samples = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get sample
            sample = test_dataset[i]
            text_embedding = sample['text_embedding'].unsqueeze(0).to(config.DEVICE)
            
            # Generate
            generated_ids = model.generate(
                text_embedding,
                max_length=config.MAX_LENGTH,
                num_beams=5
            )
            
            # Decode
            generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            original_code = test_dataset.df.iloc[i]['code']
            caption = test_dataset.df.iloc[i]['caption']
            
            samples.append({
                'caption': caption,
                'generated': generated_code,
                'original': original_code
            })
            
            print(f"\n{'='*60}")
            print(f"SAMPLE {i+1}")
            print(f"{'='*60}")
            print(f"\nCaption:\n{caption}\n")
            print(f"\nGenerated:\n{generated_code}\n")
            print(f"\nOriginal:\n{original_code}\n")
    
    return samples


# =============================================================================
# SECTION 8: MAIN EXECUTION
# =============================================================================

def load_dataset(file_path):
    """Load Mermaid dataset"""
    with pa.OSFile(file_path, 'rb') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
    return table.to_pandas()


def split_dataset(df, train_split=0.8, val_split=0.1, seed=42):
    """Split dataset"""
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


def main():
    """Main execution"""
    
    print("=" * 80)
    print("PHASE 2: COMPLETE CAPTION-TO-MERMAID TRAINING")
    print("=" * 80)
    
    # Step 1: Load dataset
    df = load_dataset(config.DATA_PATH)
    splits = split_dataset(df, config.TRAIN_SPLIT, config.VAL_SPLIT)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    # Step 2: Load Phase 1 components
    text_encoder = SentenceTransformer(config.TEXT_ENCODER)
    projection_module = load_phase1_model(config.PHASE1_MODEL_PATH, config)
    
    # Step 3: Initialize decoder
    decoder = MermaidDecoder(
        graph_word_dim=config.GRAPH_WORD_DIM,
        num_graph_words=config.NUM_GRAPH_WORDS,
        gpt2_model=config.GPT2_MODEL
    )
    
    # Step 4: Create complete model
    complete_model = CaptionToMermaidModel(projection_module, decoder)
    
    total_params = sum(p.numel() for p in complete_model.parameters())
    print(f"\n✓ Complete model initialized")
    print(f"  Total parameters: {total_params:,}")
    
    # Step 5: Create datasets
    train_dataset = MermaidDatasetPhase2(
        splits['train'], text_encoder, decoder.tokenizer, config.MAX_LENGTH
    )
    val_dataset = MermaidDatasetPhase2(
        splits['val'], text_encoder, decoder.tokenizer, config.MAX_LENGTH
    )
    test_dataset = MermaidDatasetPhase2(
        splits['test'], text_encoder, decoder.tokenizer, config.MAX_LENGTH
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
    
    # Step 7: Train!
    history = train_phase2(complete_model, train_loader, val_loader, config)
    
    # Step 8: Generate samples
    samples = generate_samples(complete_model, test_dataset, decoder.tokenizer, num_samples=5)
    
    # Save samples
    with open(os.path.join(config.OUTPUT_DIR, 'generated_samples.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PHASE 2 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {config.MODEL_DIR}/best_complete_model.pt")
    print(f"Samples saved to: {config.OUTPUT_DIR}/generated_samples.json")


if __name__ == "__main__":
    main()
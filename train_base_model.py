#!/usr/bin/env python3
"""
train_base_model.py — Train base MentalBERT model on large local subset before FL

Option B: Train on large local dataset first, then use as initialization for FL.
This provides a better starting point than random initialization.

Usage:
    python train_base_model.py --data_dir ./data --epochs 3 --batch_size 16
    python train_base_model.py --data_dir ./data --epochs 3 --use_mentalbert
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("base_model_training.log")
    ]
)
log = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MENTALBERT_PATH = Path.home() / ".federated" / "models" / "mentalbert"
OUTPUT_DIR = Path.home() / ".federated" / "models" / "base_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SECTION 1: DATASET LOADING (DAIC-WOZ format)
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for base model training."""
    data_dir: str
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    test_split: float = 0.2
    seed: int = 42
    use_mentalbert: bool = True
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_dir: str = str(OUTPUT_DIR)


class DepressionDataset(Dataset):
    """Dataset for depression classification."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_daic_dataset(data_dir: Path) -> Tuple[List[str], List[int]]:
    """
    Load DAIC-WOZ dataset.
    Expects:
        - data/labels.csv with Participant_ID, PHQ8_Binary columns
        - data/{ID}_P/features/{ID}_Transcript.csv or text files
    """
    log.info(f"Loading dataset from {data_dir}")
    
    # Load labels
    labels_file = data_dir / "labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels_df = pd.read_csv(labels_file)
    
    # Determine label column
    if 'PHQ8_Binary' in labels_df.columns:
        label_col = 'PHQ8_Binary'
    elif 'PHQ8_Score' in labels_df.columns:
        # Convert score to binary (threshold = 10)
        labels_df['PHQ8_Binary'] = (labels_df['PHQ8_Score'] >= 10).astype(int)
        label_col = 'PHQ8_Binary'
    else:
        raise ValueError("No PHQ8_Binary or PHQ8_Score column found in labels.csv")
    
    # Create mapping from participant ID to label
    id_to_label = dict(zip(
        labels_df['Participant_ID'].astype(str),
        labels_df[label_col].astype(int)
    ))
    
    texts = []
    labels = []
    
    # Find all participant directories
    participant_dirs = sorted([d for d in data_dir.iterdir() 
                               if d.is_dir() and d.name.startswith(tuple('0123456789'))])
    
    log.info(f"Found {len(participant_dirs)} participant directories")
    
    for pdir in participant_dirs:
        pid = pdir.name.split('_')[0]  # Extract ID from "123_P" format
        
        if pid not in id_to_label:
            log.warning(f"No label found for participant {pid}, skipping")
            continue
        
        # Find transcript file
        transcript_file = None
        features_dir = pdir / "features"
        
        if features_dir.exists():
            # Look for transcript in features directory
            for f in features_dir.glob("*Transcript*"):
                if f.suffix.lower() in ['.csv', '.txt']:
                    transcript_file = f
                    break
        
        if transcript_file is None:
            # Look in participant directory
            for f in pdir.glob("*Transcript*"):
                if f.suffix.lower() in ['.csv', '.txt']:
                    transcript_file = f
                    break
        
        if transcript_file is None:
            log.warning(f"No transcript found for participant {pid}, skipping")
            continue
        
        # Extract text from transcript
        try:
            text_content = extract_text_from_transcript(transcript_file)
            if text_content.strip():
                texts.append(text_content)
                labels.append(id_to_label[pid])
        except Exception as e:
            log.error(f"Error reading transcript for {pid}: {e}")
    
    log.info(f"Loaded {len(texts)} samples")
    log.info(f"Class distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return texts, labels


def extract_text_from_transcript(file_path: Path) -> str:
    """Extract participant utterances from transcript file."""
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        
        # Find text column
        text_col = None
        for col in ['value', 'text', 'content', 'utterance']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            text_col = df.columns[-1]  # Last column as fallback
        
        # Find speaker column if exists
        speaker_col = None
        for col in ['speaker', 'role', 'label']:
            if col in df.columns:
                speaker_col = col
                break
        
        if speaker_col:
            # Filter for participant/patient utterances only
            mask = df[speaker_col].astype(str).str.lower().str.contains(
                r'participant|patient|\bp\b', na=False
            )
            df = df[mask]
        
        # Concatenate all utterances
        texts = df[text_col].dropna().astype(str).tolist()
        return " ".join(texts)
    
    else:  # .txt file
        return file_path.read_text(encoding='utf-8')


# ============================================================================
# SECTION 2: MODEL DEFINITION
# ============================================================================

class DepressionClassifier(nn.Module):
    """MentalBERT-based classifier for depression detection."""
    
    def __init__(self, model_name_or_path: str, num_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {'loss': loss, 'logits': logits}


# ============================================================================
# SECTION 3: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            
            probs = F.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Calculate AUC if we have both classes
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_base_model(config: TrainingConfig):
    """Main training function."""
    
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    log.info(f"Using device: {DEVICE}")
    log.info(f"Configuration: {asdict(config)}")
    
    # Load data
    data_dir = Path(config.data_dir)
    texts, labels = load_daic_dataset(data_dir)
    
    if len(texts) == 0:
        raise ValueError("No data loaded. Check your data directory structure.")
    
    # Split data
    dataset = DepressionDataset(
        texts=texts,
        labels=labels,
        tokenizer=None,  # Will initialize tokenizer below
        max_length=config.max_length
    )
    
    # Initialize tokenizer
    model_path = str(MENTALBERT_PATH) if config.use_mentalbert and MENTALBERT_PATH.exists() else "bert-base-uncased"
    log.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset.tokenizer = tokenizer
    
    # Train/val split
    val_size = int(len(dataset) * config.test_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    log.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    log.info(f"Loading model from: {model_path}")
    model = DepressionClassifier(model_path, num_classes=2)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    log.info("Starting training...")
    best_val_f1 = 0.0
    best_model_state = None
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }
    
    for epoch in range(config.epochs):
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch + 1}/{config.epochs}")
        log.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        # Log results
        log.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        log.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['train_f1'].append(train_f1)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_acc'].append(val_metrics['accuracy'])
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_auc'].append(val_metrics['auc'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info(f"✓ New best model (F1: {best_val_f1:.4f})")
    
    # Final evaluation on validation set
    log.info(f"\n{'='*60}")
    log.info("Training complete!")
    log.info(f"Best validation F1: {best_val_f1:.4f}")
    log.info(f"{'='*60}")
    
    # Save model
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_save_path = save_path / "base_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'training_history': training_history,
        'best_val_f1': best_val_f1,
        'tokenizer_name': model_path
    }, model_save_path)
    log.info(f"Model saved to: {model_save_path}")
    
    # Save tokenizer
    tokenizer_save_path = save_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_save_path)
    log.info(f"Tokenizer saved to: {tokenizer_save_path}")
    
    # Save training metrics
    metrics_path = save_path / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': asdict(config),
            'training_history': training_history,
            'best_val_f1': best_val_f1,
            'final_metrics': val_metrics
        }, f, indent=2)
    log.info(f"Metrics saved to: {metrics_path}")
    
    # Print classification report
    log.info("\nClassification Report:")
    report = classification_report(
        val_metrics['labels'],
        val_metrics['predictions'],
        target_names=['Not Depressed', 'Depressed'],
        output_dict=False
    )
    log.info(report)
    
    return {
        'model_path': str(model_save_path),
        'tokenizer_path': str(tokenizer_save_path),
        'metrics_path': str(metrics_path),
        'best_val_f1': best_val_f1,
        'config': asdict(config)
    }


def main():
    parser = argparse.ArgumentParser(description="Train base MentalBERT model for depression detection")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to DAIC-WOZ data directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="Validation/test split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_mentalbert", action="store_true", default=True,
                       help="Use MentalBERT as base model (default: True)")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUT_DIR),
                       help="Directory to save trained model")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        test_split=args.test_split,
        seed=args.seed,
        use_mentalbert=args.use_mentalbert,
        save_dir=args.save_dir,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )
    
    try:
        results = train_base_model(config)
        log.info("\n" + "="*60)
        log.info("TRAINING SUCCESSFUL!")
        log.info("="*60)
        log.info(f"Model saved to: {results['model_path']}")
        log.info(f"Tokenizer saved to: {results['tokenizer_path']}")
        log.info(f"Best validation F1: {results['best_val_f1']:.4f}")
        log.info("\nNext steps:")
        log.info("1. Copy the trained model to the server for FL initialization")
        log.info("2. Update the orchestrator to use this model as the initial global model")
        log.info("3. Start federated learning rounds with clients")
        
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
TruthLens Deep Learning Model Trainer (PyTorch)
Trains Conv1D + BiLSTM + Multi-Head Self-Attention + Multi-Layer Dense NN
with AdamW optimizer, Backpropagation, and Validation Metrics.
"""

import os
import sys
import time
import json
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from dl_model import LightFakeNewsDL, SimpleTokenizer, MODEL_DIR
from dataset_downloader import build_real_benchmark_dataset, DATASET_DIR


class NewsDataset(Dataset):
    """PyTorch Dataset wrapper for sequence tensors."""

    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_pytorch_dl_model(epochs: int = 4, batch_size: int = 32, lr: float = 1e-3, sample_mode: bool = True):
    print("=" * 70)
    print("  TruthLens PyTorch Deep Learning Model Training")
    print("=" * 70)

    processed_csv = os.path.join(DATASET_DIR, "processed_train_data.csv")
    if not os.path.exists(processed_csv):
        print("[*] Training data not found. Running dataset downloader...")
        processed_csv = build_real_benchmark_dataset(sample_mode=sample_mode, max_rows=10000)

    df = pd.read_csv(processed_csv)
    print(f"[1/5] Loaded dataset: {len(df):,} records")

    # Balance Real (0) and Fake (1) classes
    df_real = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]
    min_count = min(len(df_real), len(df_fake))
    # Oversample / sample to balance
    balanced_real = df_real.sample(n=min(len(df_real), min_count * 2), random_state=42)
    balanced_fake = df_fake.sample(n=len(df_fake), random_state=42)
    # If fake is smaller, duplicate fake records to achieve 50:50 ratio
    if len(balanced_fake) < len(balanced_real):
        multiplier = (len(balanced_real) // len(balanced_fake)) + 1
        balanced_fake = pd.concat([balanced_fake] * multiplier, ignore_index=True).iloc[:len(balanced_real)]

    df_balanced = pd.concat([balanced_real, balanced_fake], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"      Balanced Training Subset: {len(df_balanced):,} records (Real: {(df_balanced['label']==0).sum():,}, Fake: {(df_balanced['label']==1).sum():,})")

    texts = df_balanced['full_text'].astype(str).tolist()
    labels = df_balanced['label'].astype(int).tolist()

    # 2. Build Vocabulary & Tokenize
    print("[2/5] Building word vocabulary & converting sequences...")
    tokenizer = SimpleTokenizer(max_words=25000, max_len=128)
    tokenizer.fit_on_texts(texts)

    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"      Saved vocabulary ({len(tokenizer.word2idx):,} words) to {vocab_path}")

    sequences = [tokenizer.text_to_sequence(t) for t in texts]

    # 3. Train / Validation Split
    print("[3/5] Splitting train (80%) and validation (20%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.20, random_state=42, stratify=labels
    )

    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Target Compute Device: {device}")

    # 4. Instantiate PyTorch DL Architecture
    vocab_size = len(tokenizer.word2idx)
    model = LightFakeNewsDL(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n[4/5] Training Deep Learning Model (CNN + BiLSTM + Attention + Backpropagation)...")
    t0 = time.time()

    best_val_acc = 0.0
    model_save_path = os.path.join(MODEL_DIR, "light_dl_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()  # Backpropagation - calculate gradients of weights and biases

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Loop
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1]

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)

        print(f"  Epoch [{epoch}/{epochs}] — Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"   Saved best PyTorch model checkpoint (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - t0
    print(f"\n[5/5] Finalizing Model Evaluation & Metrics ({total_time:.1f}s)...")

    final_roc_auc = roc_auc_score(val_targets, val_probs)
    report = classification_report(val_targets, val_preds, target_names=['REAL', 'FAKE'])

    print("\n" + "=" * 70)
    print("  [SUCCESS] PYTORCH DEEP LEARNING TRAINING COMPLETE")
    print(f"     * Best Validation Accuracy : {best_val_acc*100:.2f}%")
    print(f"     * ROC-AUC Score            : {final_roc_auc:.4f}")
    print(f"     * Saved Model Checkpoint  : {model_save_path}")
    print("=" * 70)
    print(report)

    # Save metrics JSON report
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "val_accuracy": round(float(best_val_acc), 4),
            "roc_auc": round(float(final_roc_auc), 4),
            "epochs": epochs,
            "vocab_size": vocab_size,
            "architecture": "Conv1D + BiLSTM + Multi-Head Self-Attention + GELU/Swish Dense NN"
        }, f, indent=2)


if __name__ == "__main__":
    sample = '--full' not in sys.argv
    train_pytorch_dl_model(epochs=4, batch_size=32, lr=1e-3, sample_mode=sample)

"""
Project 45: Character-Level Language Model (Char-RNN)
=====================================================
This project trains an autoregressive character-level recurrent language model
on the complete Tiny Shakespeare corpus for next-character prediction and text generation.

Key Features:
- Character tokenization with compact vocabulary (~65 unique symbols)
- Multi-layer recurrent backbone (LSTM/RNN) with hidden state tracking
- Autoregressive next-character modeling with Cross-Entropy Loss
- Perplexity (PPL) and Bits-per-Character (BPC) evaluation metrics
- Temperature-conditioned stochastic sampling (T = 0.2, 0.7, 1.2)
- Publication-quality diagnostic visualization (Loss, PPL/BPC, Bigram transition heatmap, Generation dynamics)
- Interactive text generation showcase from classical dramatic prompts
"""

import os
import sys
import time
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================================
# 1. Dataset Loading & Character Tokenizer
# =====================================================================

class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.idx2char[i] for i in indices])

    def __len__(self):
        return self.vocab_size


class CharSeqDataset(Dataset):
    """Samples contiguous character sequence chunks of length seq_len."""
    def __init__(self, data_tensor: torch.Tensor, seq_len: int = 100, num_samples: int = 3000):
        self.data = data_tensor
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.max_idx = len(data_tensor) - seq_len - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = random.randint(0, self.max_idx)
        chunk = self.data[start : start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_corpus(filepath: str, seq_len: int = 100) -> Tuple[CharTokenizer, DataLoader, DataLoader, str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing corpus file at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[Corpus] Loaded {len(text):,} total characters.")
    tokenizer = CharTokenizer(text)
    print(f"[Tokenizer] Vocabulary: {tokenizer.vocab_size} unique characters: {repr(''.join(tokenizer.chars))}")

    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    train_dataset = CharSeqDataset(train_data, seq_len=seq_len, num_samples=3500)
    val_dataset = CharSeqDataset(val_data, seq_len=seq_len, num_samples=600)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return tokenizer, train_loader, val_loader, text


# =====================================================================
# 2. Autoregressive Character-Level Language Model Architecture
# =====================================================================

class CharRNNLM(nn.Module):
    """
    Multi-layer autoregressive character-level recurrent language model.
    Features:
    - Dense character embedding lookup
    - Multi-layer recurrent backbone (LSTM with cell and hidden state)
    - Inter-layer dropout
    - Linear projection to next-character vocabulary distribution
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch_size, seq_len)
        embeds = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embeds, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=DEVICE)
        return (h0, c0)


# =====================================================================
# 3. Temperature-Conditioned Text Generation
# =====================================================================

def generate_text(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str = "ROMEO:\n",
    max_len: int = 250,
    temperature: float = 0.8
) -> str:
    """
    Generates text character-by-character conditioned on prompt and sampling temperature.
    Temperature T controls stochastic entropy:
    - T < 0.5: Highly conservative, deterministic
    - T ~ 0.7 - 0.9: Fluent, balanced creativity
    - T > 1.0: Diverse, high-entropy
    """
    model.eval()
    if not prompt:
        prompt = "A"

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.char2idx.get("A", 0)]

    input_seq = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    hidden = None

    # Warmup hidden state with prompt
    with torch.no_grad():
        logits, hidden = model(input_seq, hidden)

    # Autoregressive generation loop
    generated = list(tokens)
    current_char = torch.tensor([[tokens[-1]]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(current_char, hidden)
            # Logits shape: (1, 1, vocab_size) -> (vocab_size,)
            last_logits = logits.squeeze(0).squeeze(0) / max(temperature, 1e-5)
            probs = torch.softmax(last_logits, dim=-1)

            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_char_idx)
            current_char = torch.tensor([[next_char_idx]], dtype=torch.long, device=DEVICE)

    return tokenizer.decode(generated)


# =====================================================================
# 4. Training & Validation Pipeline
# =====================================================================

def evaluate(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    perplexity = math.exp(min(avg_loss, 20.0))
    bits_per_char = avg_loss / math.log(2.0)
    return avg_loss, perplexity, bits_per_char


def train_char_rnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    epochs: int = 8,
    lr: float = 2e-3
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "perplexity": [], "bpc": [], "epoch_times": []
    }

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "char_rnn_best.pt")

    print(f"\n[Training] Training Char-RNN Language Model for {epochs} epochs on {DEVICE}...")
    start_total = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_ppl, val_bpc = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["perplexity"].append(val_ppl)
        history["bpc"].append(val_bpc)
        history["epoch_times"].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "perplexity": val_ppl,
                "bpc": val_bpc
            }, checkpoint_path)

        print(f"Epoch {epoch:02d}/{epochs:02d} ({epoch_time:.2f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Perplexity: {val_ppl:.2f} | "
              f"Bits/Char (BPC): {val_bpc:.2f}")

        # Intermediate sample generation
        if epoch in (1, epochs // 2, epochs):
            sample = generate_text(model, tokenizer, prompt="ROMEO:\n", max_len=120, temperature=0.7)
            clean_sample = sample.replace("\n", " ")
            print(f"  [Sample Gen E{epoch}]: \"{clean_sample[:110]}...\"")

    total_time = time.time() - start_total
    print(f"\n[Training] Completed in {total_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")
    return history


# =====================================================================
# 5. Diagnostic Visualizations
# =====================================================================

def plot_diagnostics(
    history: Dict[str, List[float]],
    tokenizer: CharTokenizer,
    raw_text: str,
    model: nn.Module,
    save_path: str
):
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2.5, markersize=5, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", linewidth=2.5, markersize=5, label="Val Loss")
    ax1.set_title("Cross-Entropy Loss Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (nats/char)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Perplexity & Bits-Per-Character (BPC)
    ax2 = axes[0, 1]
    color1 = "#2ca02c"
    color2 = "#d62728"
    ax2.plot(epochs, history["perplexity"], "o-", color=color1, linewidth=2.5, markersize=5, label="Perplexity (PPL)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Perplexity (exp(Loss))", color=color1, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color1)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, history["bpc"], "s--", color=color2, linewidth=2.0, markersize=5, label="Bits/Char (BPC)")
    ax2_twin.set_ylabel("Bits Per Character (BPC)", color=color2, fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor=color2)
    ax2.set_title("Perplexity & Information Density (BPC)", fontsize=13, fontweight="bold", pad=10)

    # Panel 3: Character Bigram Transition Matrix (Empirical vs Learned)
    ax3 = axes[1, 0]
    top_chars = [" ", "e", "t", "a", "o", "i", "n", "s", "h", "r", "d", "l", "u", "m", "w", "c"]
    char_indices = [tokenizer.char2idx[c] for c in top_chars if c in tokenizer.char2idx]
    
    # Compute empirical transition count matrix from raw text
    trans_matrix = np.zeros((len(char_indices), len(char_indices)))
    text_sample = raw_text[:50000]
    for i in range(len(text_sample) - 1):
        c1, c2 = text_sample[i], text_sample[i+1]
        if c1 in top_chars and c2 in top_chars:
            idx1 = top_chars.index(c1)
            idx2 = top_chars.index(c2)
            trans_matrix[idx1, idx2] += 1

    trans_norm = trans_matrix / (trans_matrix.sum(axis=1, keepdims=True) + 1e-9)

    im = ax3.imshow(trans_norm, cmap="Blues", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability P(c_{t+1} | c_t)", fontsize=10)

    labels = ["' '" if c == " " else f"'{c}'" for c in top_chars]
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_title("Top Character Transition Matrix (Corpus Phonotactics)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Next Character (t+1)", fontsize=11)
    ax3.set_ylabel("Current Character (t)", fontsize=11)
    ax3.grid(False)

    # Panel 4: Temperature-Dependent Generation Comparison
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Temperature-Conditioned Text Generation Showcase", fontsize=13, fontweight="bold", pad=10)

    t_low = generate_text(model, tokenizer, prompt="KING:\n", max_len=140, temperature=0.2)
    t_med = generate_text(model, tokenizer, prompt="KING:\n", max_len=140, temperature=0.7)
    t_high = generate_text(model, tokenizer, prompt="KING:\n", max_len=140, temperature=1.2)

    def clean(txt):
        return txt.replace("\n", " ").strip()

    display_text = (
        f"Temperature T = 0.2 (Low Entropy / Conservative):\n"
        f"\"{clean(t_low)[:130]}...\"\n\n"
        f"Temperature T = 0.7 (Balanced / Coherent):\n"
        f"\"{clean(t_med)[:130]}...\"\n\n"
        f"Temperature T = 1.2 (High Entropy / Creative):\n"
        f"\"{clean(t_high)[:130]}...\""
    )

    ax4.text(
        0.02, 0.95, display_text,
        transform=ax4.transAxes,
        fontsize=9.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95)
    )

    plt.suptitle("Project 45: Character-Level Recurrent Language Model (Tiny Shakespeare)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Diagnostic chart saved to: {save_path}")


# =====================================================================
# 6. Main Pipeline
# =====================================================================

def main():
    print("=" * 70)
    print("Project 45: Character-Level Recurrent Language Model (Char-RNN)")
    print("=" * 70)

    # 1. Load Corpus & Tokenizer
    corpus_path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    tokenizer, train_loader, val_loader, raw_text = load_corpus(corpus_path, seq_len=100)

    # 2. Instantiate Model
    model = CharRNNLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] CharRNNLM initialized with {num_params:,} trainable parameters.")

    # 3. Model Training
    history = train_char_rnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=8,
        lr=2e-3
    )

    # 4. Final Evaluation
    loss_fn = nn.CrossEntropyLoss()
    final_loss, final_ppl, final_bpc = evaluate(model, val_loader, loss_fn)
    print("\n" + "=" * 70)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 70)
    print(f"Validation Loss:        {final_loss:.4f} nats/char")
    print(f"Validation Perplexity:  {final_ppl:.2f}")
    print(f"Bits Per Character:     {final_bpc:.2f} BPC")

    # 5. Diagnostic Visualizations
    vis_path = os.path.join(PROJECT_DIR, "char_rnn_results.png")
    plot_diagnostics(history, tokenizer, raw_text, model, vis_path)

    # 6. Comprehensive Generation Showcase Across Classic Prompts
    print("\n" + "=" * 70)
    print("INTERACTIVE TEXT GENERATION SHOWCASE ACROSS PROMPTS & TEMPERATURES")
    print("=" * 70)

    test_prompts = [
        ("HAMLET:\nTo be, or not to be", 0.7),
        ("ROMEO:\nO, speak again, bright angel!", 0.6),
        ("GLOUCESTER:\nNow is the winter of our discontent", 0.8),
        ("First Citizen:\nWe are accounted poor citizens", 0.5)
    ]

    for prompt, temp in test_prompts:
        generated = generate_text(model, tokenizer, prompt=prompt, max_len=200, temperature=temp)
        print(f"\n--- [Prompt: {repr(prompt)} | Temp: {temp}] ---")
        print(generated)
        print("-" * 50)

    print("\n[Done] Project 45 execution completed successfully!")


if __name__ == "__main__":
    main()


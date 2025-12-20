import torch
from torch.utils.data import DataLoader
from training.dataset import TokenDataset
from model.gpt_model import GPTModel
import os

# -----------------------------
# CONFIG
# -----------------------------
VOCAB_SIZE = 5000
SEQ_LEN = 512
BATCH_SIZE = 8        # keep small for CPU
EPOCHS = 5
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKEN_FILE = "data/tokenizer/tokenized_dataset.txt"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# DATASET
# -----------------------------
dataset = TokenDataset(
    token_file=TOKEN_FILE,
    seq_len=SEQ_LEN
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print(f"Dataset samples: {len(dataset)}")

# -----------------------------
# MODEL
# -----------------------------
model = GPTModel(
    vocab_size=VOCAB_SIZE,
    dim=512,
    num_heads=8,
    num_layers=8,
    ff_dim=2048,
    max_seq_len=SEQ_LEN
).to(DEVICE)

print("Model initialized on", DEVICE)

# -----------------------------
# OPTIMIZER + LOSS
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# TRAINING LOOP
# -----------------------------
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for step, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)  # (B, T, vocab)
        loss = criterion(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"gpt_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}\n")

print("🔥 Training finished")

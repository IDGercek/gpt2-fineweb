import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_info
from transformers import AutoTokenizer
from dataclasses import dataclass
import time

from model import GPT, GPTConfig

## -------- Setup --------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

@dataclass
class TrainingConfig:
    # Data
    ds_proportion: float = 0.00001
    num_workers: int = 2
    prefetch_factor: int = 2

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    ignore_index: int = -100
    

config = TrainingConfig()

## -------- Data --------

# Load the dataset and metadata
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", columns=["text"], streaming=True)
info = get_dataset_config_info("HuggingFaceFW/fineweb", "sample-10BT")

# Take a spesific proportion
ds_samples = int(info.splits["train"].num_examples * config.ds_proportion)
dataset = dataset.take(ds_samples)
print(f"Dataset initialized with {ds_samples} samples.")

# Tokenization on the fly
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=model_config.block_size,
    )

    tokens["labels"] = [ [config.ignore_index if tok == tokenizer.pad_token_id else tok for tok in seq] for seq in tokens["input_ids"] ]

    return tokens

dataset = dataset.map(tokenize, batched=True)

dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor
)

## -------- Model --------

model_config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    hidden_size= 4 * 768,
    dropout=0.2
)
model = GPT(model_config).to(device)
parameter_count = sum([p.numel() for p in model.parameters()])
print(f"Model initialized with {'{:,}'.format(parameter_count)} parameters")

criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

## -------- Training --------

epochs = 1
max_steps = epochs * ds_samples
epoch_print_interval = 10
print(f"Training for {max_steps} steps:")

for step, batch in enumerate(dataloader):
    t0 = time.time() # Start time

    # Prepare data
    # Data comes in shape (max_len, batch_size). We convert it to (batch_size, max_len) with torch.stack
    input_ids = torch.stack(batch["input_ids"], dim=1).to(device)
    labels = torch.stack(batch["labels"], dim=1).to(device)
    attn_mask = torch.stack(batch["attention_mask"], dim=1).to(device)
    attn_mask = attn_mask == 1 # convert from int to bool

    t1 = time.time() # Data preparation time

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(input_ids, attn_mask)

    # Backpropagation
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    # Time calculation
    torch.cuda.synchronize()
    t2 = time.time()
    data_prep = (t1 - t0) * 1000
    step_time = (t2 - t1) * 1000

    if step % epoch_print_interval == 0:
        print(f"Step: {step} | Loss: {loss.item():.4f} | Data Prep: {data_prep:.2f}ms | Step Time: {step_time:.2f}ms")


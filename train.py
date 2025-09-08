import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_info
from transformers import AutoTokenizer
from dataclasses import dataclass
import math
import time
import os
from google.colab import drive
from model import GPT, GPTConfig

## -------- Setup --------

drive.mount('/content/drive')
os.mkdir("/content/drive/My Drive/gpt2-fineweb")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingConfig:
    # Data
    ds_proportion: float = 0.002
    num_workers: int = 2
    prefetch_factor: int = 4

    # Training
    batch_size: int = 6
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 6e-5
    ignore_index: int = -100
    

config = TrainingConfig()

model_config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    hidden_size= 4 * 768,
    dropout=0.2
)

## -------- Data --------

# Load the dataset and metadata
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", columns=["text"], streaming=True)
info = get_dataset_config_info("HuggingFaceFW/fineweb", "sample-10BT")

# Take a spesific proportion
ds_samples = int(info.splits["train"].num_examples * config.ds_proportion)
dataset = dataset.take(ds_samples)
print(f"Dataset initialized with {ds_samples} samples.")

# Shuffle dataset
dataset = dataset.shuffle(buffer_size=100_000)

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

dataloader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        prefetch_factor=config.prefetch_factor,
                        drop_last=True,
                        pin_memory=True,
                        persistent_workers=True
)

## -------- Model --------

model = GPT(model_config).to(device)
parameter_count = sum([p.numel() for p in model.parameters()])
print(f"Model initialized with {'{:,}'.format(parameter_count)} parameters")

criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)
scaler = torch.amp.GradScaler(device, enabled=True)

## -------- Training --------

# Learning rate scheduling with cosine decay
max_steps = ds_samples // config.batch_size
warmup_steps = max_steps // 20
def get_lr(step):
    if step < warmup_steps: # Warmup
        return config.max_learning_rate * (step + 1) / warmup_steps
    if step > max_steps: # After max_steps, to prevent errors
        return config.min_learning_rate

    # In between, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.max_learning_rate - config.min_learning_rate)

losses = []
epoch_print_interval = 100
checkpoint_interval = 1000
print(f"Training for maximum {max_steps} steps:")

model.train()

t = time.time() # Start time
for step, batch in enumerate(dataloader):
    # Prepare data
    # Data comes in shape (max_len, batch_size). We convert it to (batch_size, max_len) with torch.stack
    input_ids = torch.stack(batch["input_ids"], dim=1).to(device)
    labels = torch.stack(batch["labels"], dim=1).to(device)
    key_padding_mask = torch.stack(batch["attention_mask"], dim=1).to(device)
    key_padding_mask = key_padding_mask == 0 # convert from int to bool

    # Zero gradients
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
        # Forward pass
        logits = model(input_ids, key_padding_mask=key_padding_mask)

        # Loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses.append(loss.item())

    # Backpropagation
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping

    scaler.step(optimizer)
    scaler.update()

    if step % epoch_print_interval == 0:
        # Time calculation
        torch.cuda.synchronize()
        average_step_time = (time.time() - t) / epoch_print_interval # in seconds
        t = time.time()

        tokens_per_second = config.batch_size * model_config.block_size // average_step_time

        print(f"Step: {step} | Loss: {loss.item():.4f} | Lr: {lr:.4e} | Norm: {norm:.4f} | Avg Step Time: {average_step_time*1000:.2f}ms | {tokens_per_second} tokens/sec")

    if step % checkpoint_interval == 0 and step != 0:
      checkpoint = {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()}
      torch.save(checkpoint, f"/content/drive/My Drive/gpt2-fineweb/checkpoint_{step}.pt")
      print(f"Saved checkpoint to checkpoint_{step}.pt")

print(f"Training completed after {step} steps.")

# Save final model
checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
torch.save(checkpoint, "/content/drive/My Drive/gpt2-fineweb/checkpoint_final.pt")
print("Saved checkpoint to checkpoint_final.pt")

# Save losses
torch.save(losses, "content/drive/My Drive/gpt2-fineweb/losses.pt")
print("Saved losses to losses.pt")

## -------- Inference --------

def generate(input, num_return_sequences, max_length):
    tokens = tokenizer.encode(input)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indicies, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f"{i} | {decoded}")
        print("----")

print("Running inference...")
generate("I'm a language model", 3, 32)
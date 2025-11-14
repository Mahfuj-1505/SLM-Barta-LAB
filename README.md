# SLM-Barta-LAB - StoryTeller AI Model

A GPT-style Small Language Model (SLM) trained on the TinyStories dataset for generating creative short stories. This model implements a transformer-based architecture with 30M parameters.

## 📋 Table of Contents
- [Model Overview](#model-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Model Configuration](#model-configuration)
- [Results](#results)

## 🎯 Model Overview

**StoryTeller** is a GPT-based language model designed for generating children's stories. The model features:
- **Parameters**: ~30M (29,995,392 trainable parameters)
- **Architecture**: 6-layer transformer with 6 attention heads
- **Context Window**: 128 tokens
- **Vocabulary Size**: 50,257 tokens (GPT-2 tokenizer)
- **Training Data**: TinyStories dataset (~472M tokens)

## 📦 Requirements

```bash
torch>=2.0.0
tiktoken
numpy
tqdm
matplotlib
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mahfuj-1505/SLM-Barta-LAB.git
cd SLM-Barta-LAB
```

### 2. Install Dependencies
```bash
pip install torch tiktoken numpy tqdm matplotlib
```

**You can skip these steps and take the trained model and use it.**
To do so check the [Inference](#inference) section.

### 3. Download Dataset 
Download the TinyStories dataset in binary format:
- `train.bin` - Training data (~472M tokens)
- `validation.bin` - Validation data (~4.7M tokens)

## 🏗️ Model Architecture

### GPT Configuration

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257      # GPT-2 tokenizer vocabulary
    block_size: int = 128        # Maximum sequence length
    n_layer: int = 6             # Number of transformer layers
    n_head: int = 6              # Number of attention heads
    n_embd: int = 384            # Embedding dimension
    dropout: float = 0.1         # Dropout rate
    bias: bool = True            # Use bias in linear layers
```

### Model Components

1. **Token & Position Embeddings**: Maps tokens and positions to embedding space
2. **Transformer Blocks**: 6 layers with:
   - Multi-head causal self-attention (6 heads)
   - Feed-forward network (4x embedding dimension)
   - Layer normalization
   - Residual connections
3. **Language Model Head**: Projects to vocabulary logits

## 📊 Dataset

The model is trained on the **TinyStories** dataset, which contains simple short stories suitable for children.

### Data Statistics
- **Training tokens**: 471,872,517
- **Validation tokens**: 4,743,928
- **Format**: Binary encoded with GPT-2 tokenizer (uint16)

### Loading Data

```python
import numpy as np

dtype = np.uint16
train_path = "path/to/train.bin"
val_path = "path/to/validation.bin"

train_data = np.memmap(train_path, dtype=dtype, mode='r')
val_data = np.memmap(val_path, dtype=dtype, mode='r')
```

## 🎓 Training

### Training Configuration

```python
# Hyperparameters
learning_rate = 1e-4           # Initial learning rate
max_iters = 20000              # Total training iterations
warmup_steps = 1000            # Learning rate warmup steps
min_lr = 5e-4                  # Minimum learning rate
batch_size = 32                # Batch size
block_size = 128               # Sequence length
gradient_accumulation_steps = 32  # Gradient accumulation
eval_iters = 500               # Evaluation frequency
```

### Training Loop

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# Initialize model
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config)
model = model.to('cuda')

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    betas=(0.9, 0.95), 
    weight_decay=0.1, 
    eps=1e-9
)

scheduler_warmup = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

# Mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=True)

# Training loop
model.train()
for epoch in range(max_iters):
    X, y = get_batch("train")
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    if ((epoch + 1) % gradient_accumulation_steps == 0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    scheduler.step()
```

### Training Features

- ✅ **Mixed Precision Training** (BF16/FP16)
- ✅ **Gradient Accumulation** (32 steps)
- ✅ **Gradient Clipping** (max_norm=0.5)
- ✅ **Learning Rate Scheduling** (Warmup + Cosine Decay)
- ✅ **Checkpoint Saving** (every 500 iterations)
- ✅ **Best Model Tracking** (based on validation loss)

## 🔮 Inference

### Load Trained Model

```python
import torch
import tiktoken

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load model
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,  # Disable dropout for inference
    bias=True
)

model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location='cuda'))
model = model.to('cuda')
model.eval()
```

### Generate Stories

```python
def generate_story(prompt, max_tokens=200, temperature=0.8, top_k=50):
    """
    Generate a story from a prompt.
    
    Args:
        prompt (str): Starting text for the story
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (higher = more random)
        top_k (int): Number of top tokens to sample from
    
    Returns:
        str: Generated story
    """
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to('cuda')
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            tokens, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_tokens = generated[0].tolist()
    story = enc.decode(generated_tokens)
    
    return story

# Example usage
prompt = "Once upon a time, there was a little girl named Lily."
story = generate_story(prompt, max_tokens=150, temperature=0.8, top_k=50)
print(story)
```

### Advanced Generation

```python
# Creative story (higher temperature)
creative_story = generate_story(
    "A brave knight", 
    max_tokens=200, 
    temperature=1.0, 
    top_k=100
)

# Focused story (lower temperature)
focused_story = generate_story(
    "One sunny day", 
    max_tokens=150, 
    temperature=0.6, 
    top_k=40
)
```

## ⚙️ Model Configuration

### Checkpoint Management

```python
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                   train_loss_list, validation_loss_list, best_val_loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss_list': train_loss_list,
        'validation_loss_list': validation_loss_list,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, scheduler, scaler, filepath):
    """Load training checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['train_loss_list'], \
               checkpoint['validation_loss_list'], checkpoint['best_val_loss']
    else:
        return 0, [], [], float('inf')
```

## 📈 Results

### Training Performance

- **Final Training Loss**: 2.3923
- **Best Validation Loss**: 2.4045
- **Training Duration**: ~4 hours (20,000 iterations)
- **Hardware**: NVIDIA GPU with CUDA support

### Loss Progression

| Epoch | Training Loss | Validation Loss | Learning Rate |
|-------|--------------|-----------------|---------------|
| 500   | 8.9662       | 8.9730         | 0.00010       |
| 5000  | 4.2165       | 4.2195         | 0.00014       |
| 10000 | 3.1668       | 3.1769         | 0.00028       |
| 15000 | 2.6671       | 2.6703         | 0.00044       |
| 19500 | 2.3923       | 2.4045         | 0.00050       |

## 📝 Files in Repository

- `storyteller-ai.ipynb` - Complete training notebook
- `StoryTeller.pt` - Model checkpoint/weights
- `best_model_params.pt` - Best model parameters (generated during training)
- `training_checkpoint.pt` - Training checkpoint (generated during training)

## 🔧 GPU Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: Compatible with CUDA 11.x or later
- **Mixed Precision**: BF16 support recommended for faster training

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

**Mahfuj-1505**
- GitHub: [@Mahfuj-1505](https://github.com/Mahfuj-1505)

## 🙏 Acknowledgments

- TinyStories dataset creators
- OpenAI for GPT-2 tokenizer
- PyTorch team for the deep learning framework

---

**Happy Story Generation! 📚✨**

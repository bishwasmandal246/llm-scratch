import torch
import wandb
import torch.nn as nn
import numpy as np

PROJECT_NAME = 'llm-scratch'
NAME = 'training'

def generate_text(model, idx, max_new_tokens, context_len, temperature, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        # idx shape: (B, T)
        # If current context is greater than context_len 
        # truncate (from the beginning)
        idx_cond = idx[:, -context_len:]
        with torch.no_grad():
            logits = model(idx_cond) # logits shape: (batch, context_len, vocab_size)
        # focus only on the last time step
        logits = logits[:, -1, :] # shape (batch, vocab_size)

        # filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def plot_losses(iteration, train_loss, validation_loss):
    wandb.log({
        "Train Loss": train_loss,
        "Validation Loss": validation_loss,
        "Iteration": iteration
    })


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokens, tokenizer):
    decoded = tokenizer.decode(tokens.squeeze(0).tolist())
    return decoded


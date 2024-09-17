import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import cross_entropy

from utils import *
from early_stopping import EarlyStopping
from model_architecture import GPTModel
from prepare_data import create_dataloader


def pretrain_model(model, start_iter, train_loader, 
                   validation_loader, optimizer, scheduler, 
                   max_grad_norm, device, save_path, num_epochs=3):
    
    early_stopping=EarlyStopping(path=save_path)
    total_iter = 0
    for _ in range(num_epochs):
        for input, target in train_loader:
            total_iter+=1
            if total_iter > start_iter:
                print(f"Iteration: {total_iter}")
                model.train()
                optimizer.zero_grad()

                input, target = input.to(device), target.to(device)
                logits = model(input)

                loss = cross_entropy(logits.flatten(0, 1), target.flatten())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()

                if total_iter % 5000 == 0:
                    validation_loss = validation(model, validation_loader, device)

                    print(f'Iteration: {total_iter}, Train Loss: {loss}, Validation Loss: {validation_loss}')
                    plot_losses(total_iter, loss, validation_loss)

                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'iteration': total_iter
                    }

                    early_stopping(validation_loss, checkpoint, total_iter)

                    if early_stopping.early_stop: 
                        print(f'Early Stopping at iteration {total_iter}')
                        break



def validation(model, validation_loader, device):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for idx_validation, (input_validation, target_validation) in enumerate(validation_loader):
            input_validation, target_validation = input_validation.to(device), target_validation.to(device)
            logits_validation = model(input_validation)
            loss_validation = cross_entropy(logits_validation.flatten(0, 1), target_validation.flatten())
            validation_loss += loss_validation.item()
    return (validation_loss/(idx_validation+1))


def main():

    # Original GPT2 124M model configuration.
    # GPT_CONFIG_124M = {
    #     "vocab_size": 50257,
    #     "context_length": 1024,
    #     "emb_dim": 768,
    #     "num_heads": 12,
    #     "n_layers": 12,
    #     "dropout": 0.1,
    #     "qkv_bias": False
    # }

    #model-old.pth configuration
    # GPT_CONFIG_SMALL = {
    #     "vocab_size": 50257,
    #     "context_length": 512,
    #     "emb_dim": 256,
    #     "num_heads": 4,
    #     "n_layers": 2,
    #     "dropout": 0.1,
    #     "qkv_bias": False
    # }

    GPT_CONFIG_SMALL = {
        "vocab_size": 50257,
        "context_length": 512,
        "emb_dim": 512,
        "num_heads": 16,
        "n_layers": 2,
        "dropout": 0.1,
        "qkv_bias": False
    }


    train_tokens = 'data/tiny-stories/train_tokens.txt'
    validation_tokens =  'data/tiny-stories/validation_tokens.txt'

    save_model_dir = 'saved_model'
    os.makedirs(save_model_dir, exist_ok=True)

    model_name = 'model.pth'
    model_name_path = os.path.join(save_model_dir, model_name)


    torch.manual_seed(100)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA accelerator found. Using CUDA to train.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS accelerator found. Using MPS to train.")
    else:
        device = torch.device("cpu")
        print("No accelerator found. Using CPU to train.")


    model = GPTModel(GPT_CONFIG_SMALL)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Model Parameters: {total_params}')
    model.to(device)

    learning_rate = 5e-5
    batch_size = 32
    warmup_steps = 100
    max_grad_norm = 1.0

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step)/(float(max(1, warmup_steps)))
        return max(0.0, float(len(train_dataloader) - current_step) / float(max(1, len(train_dataloader) - warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    if os.path.exists(model_name_path):
        print('Previous saved model exists. Loading Weights...')
        checkpoint = torch.load(model_name_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['iteration']
        print(f'Starting training from Iteration {start_iter}')
    else:
        start_iter = -1
        print('No saved checkpoints found. Training from start...')

    train_dataloader, validation_dataloader = create_dataloader(tokenized_text_file_train=train_tokens,
                                                                tokenized_text_file_validation=validation_tokens, 
                                                                batch_size=batch_size,
                                                                context_length=GPT_CONFIG_SMALL["context_length"],
                                                                stride=GPT_CONFIG_SMALL["context_length"]//2, 
                                                                shuffle=True,
                                                                drop_last=True,
                                                                num_workers=0)
    

    wandb.require("core")
    wandb.init(project=PROJECT_NAME, name=NAME)

    pretrain_model(model=model, 
                   start_iter=start_iter, 
                   train_loader=train_dataloader,
                   validation_loader=validation_dataloader,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   max_grad_norm=max_grad_norm,
                   device=device,
                   save_path=model_name_path,
                   num_epochs=3)
    
    wandb.finish()
    

if __name__ == '__main__':
    main()

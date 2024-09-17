import torch
import numpy as np


class EarlyStopping:
    def __init__(self,
                 patience=5,
                 delta=0,
                 path='saved_model/model.pth'):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, checkpoint, idx):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, checkpoint, idx)

        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, checkpoint, idx)
            self.counter = 0

    def save_checkpoint(self, val_loss, checkpoint, idx):
        print(f'Model and optimizer saved at iteration {idx} to {self.path}')
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
        
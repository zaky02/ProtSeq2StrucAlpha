import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait for evaluation loss improvement.
            delta (float): Minimum improvement in evaluation loss to reset patience counter.
            verbose (bool): Whether to print detailed logs during execution.
        """
        self.patience = patience 
        self.delta = delta 
        self.counter = 0 
        self.best_score = np.inf
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, score, model, weights_path, fabric):
        
        if self.best_score - score > self.delta:
            self.best_score = score
            if self.verbose > 1:
                fabric.print(f"Best evaluation loss updated to: {self.best_score:.6f}")
                fabric.print("Saving model weights and reseting counter...")
            self.counter = 0
            self.save_checkpoint(model, weights_path, fabric)

        else:
            self.counter += 1
            if self.verbose > 1:
                fabric.print(f"EarlyStopping counter: {self.counter}")
                fabric.print(f"{self.counter}/{self.patience} till early stopping is enforced")
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, model, weights_path, fabric):
        state = {'model': model}
        fabric.save(weights_path, state)

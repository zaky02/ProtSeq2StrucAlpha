import torch

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
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, eval_loss, model, weights_path, fabric):
        score = -eval_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, weights_path, fabric)
            if self.verbose > 1:
                fabric.print(f'EarlyStopping: Evaluation score improved ({self.best_score:.6f} --> {score:.6f}).')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose > 1:
                fabric.print(f'EarlyStopping: EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, weights_path, fabric)
            self.counter = 0
            if self.verbose > 1:
                fabric.print(f'EarlyStopping: Evaluation score improved ({self.best_score:.6f} --> {score:.6f}).  Resetting counter to 0.')
            
    def save_checkpoint(self, model, weights_path, fabric):
        state = {'model': model}
        fabric.save(weights_path, state)

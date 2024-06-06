import numpy as np
import torch

# inspired by:
# https://github.com/Bjarten/early-stopping-pytorch
# which is in turn inspired by the PyTorch Ignite implementation of Early Stopping:
# https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping


class EarlyStopping:
    """Early stops the trainer if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10000, verbose=True, delta=0, path='checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.best_acc = None

    def __call__(self, val_loss, model, acc=0):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_acc = acc
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_acc = acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        print(f"Best acc: {self.best_acc}")

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        model.save_model_to_file(self.path)
        self.val_loss_min = val_loss

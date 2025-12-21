import torch
import copy
from src.config import MODEL_SAVE_PATH


class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a given patience.
    Replicates Keras 'restore_best_weights=True' [cite: 194-195].
    """

    def __init__(self, patience=3, verbose=False, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)
            print("Restored best model weights.")
        return model


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)
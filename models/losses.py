import warnings

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self):

        super(PearsonLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, x1, x2):

        pearson = self.cos(x1, x2)
        return -torch.mean(pearson)


class OrdinalClassificationLoss(nn.Module):
    def __init__(self):
        super(OrdinalClassificationLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, predictions, targets):

        # Create out modified target with [batch_size, num_labels] shape
        modified_target = torch.zeros_like(predictions)

        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(targets):
            modified_target[i, 0 : int(target) + 1] = 1

        return self.mse(predictions, modified_target).sum()


class ClassificationLoss(nn.Module):
    """Cross entropy loss. Regression targets are rounded and treated as class
    labels. Predictions are taken to be model logits i.e. not softmax
    probabilities

    If used, one shoud create a multi-output model and modify outputs
    according to:

    >>> y_pred = torch.argmax(nn.Softmax()(y_pred), axis=1)
    >>> val_pred = torch.argmax(nn.Softmax()(val_pred), axis=1)
    """

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):

        if torch.max(targets) <= 1:
            warnings.warn("Targets are expected to be in range [0.0, 17.0]")

        targets = torch.round(targets).long().squeeze()
        return self.ce(predictions, targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, targets):

        return torch.mean(torch.abs(predictions - targets) ** (2 + self.gamma))

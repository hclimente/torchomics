import torch
import torch.nn as nn


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

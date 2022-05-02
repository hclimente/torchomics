import random

import numpy as np
import torch


def fix_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

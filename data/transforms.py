import random

import torch
from torch.distributions.beta import Beta


class Transform:
    def __init__(self):
        pass

    def __call__(self, *args):
        if self:
            args = [x.clone() for x in args]
            return self.apply(*args)
        else:
            return args


class Mixup(Transform):
    def __init__(self, alpha, dataset):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(alpha, alpha) if alpha > 0 else None
        self.dataset = dataset

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, rc, expression):

        p = self.p_dist.sample()

        r_seq, r_rc, r_expr = item_sampler(self.dataset)
        seq = p * seq + (1 - p) * r_seq
        rc = p * rc + (1 - p) * r_rc
        expression = p * expression + (1 - p) * r_expr

        return seq, rc, expression


class Cutmix(Transform):
    def __init__(self, alpha, dataset):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(alpha, alpha) if alpha > 0 else None
        self.dataset = dataset

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, rc, expression):

        length = seq.shape[1]

        p = self.p_dist.sample()
        width = int(length * torch.sqrt(1 - p) / 2)

        pos = random.randint(0, length)
        x0, x1 = max(0, pos - width), min(length, pos + width)

        r_seq, r_rc, r_expr = item_sampler(self.dataset)
        seq[:, x0:x1] = r_seq[:, x0:x1]
        rc[:, x0:x1] = r_rc[:, x0:x1]
        expression = p * expression + (1 - p) * r_expr

        return seq, rc, expression


class RandomErase(Transform):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(alpha, alpha) if alpha > 0 else None

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, rc, expression):

        length = seq.shape[1]

        p = self.p_dist.sample()
        width = int(length * torch.sqrt(1 - p) / 2)

        pos = random.randint(0, length)
        x0, x1 = max(0, pos - width), min(length, pos + width)

        seq[:, x0:x1] = torch.rand(seq[:, x0:x1].shape)
        rc[:, x0:x1] = torch.rand(rc[:, x0:x1].shape)

        return seq, rc, expression


class Mutate(Transform):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __bool__(self):
        return self.n > 0

    def apply(self, seq, rc, expression):

        length = seq.shape[1]

        pos = torch.randint(high=length, size=(self.n,))
        perm = torch.vstack([torch.randperm(4) for _ in range(self.n)]).T

        seq[:, pos] = seq[perm, pos]
        rc[:, pos] = rc[perm, pos]

        return seq, rc, expression


def item_sampler(dataset):
    i = random.randint(0, len(dataset) - 1)

    return dataset.sequences[i], dataset.rc_sequences[i], dataset.expression[i]

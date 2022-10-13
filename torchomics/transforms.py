import random

import torch
from torch.distributions.beta import Beta


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if self:
            args = [x.clone() for x in args]
            return self.apply(*args)
        else:
            return args


class Mixup(Transform):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(1, alpha) if alpha > 0 else None

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, expression):

        p = self.p_dist.sample()

        r_seq, r_expr = seq.roll(1, 0), expression.roll(1, 0)
        seq = p * seq + (1 - p) * r_seq
        expression = p * expression + (1 - p) * r_expr

        return seq, expression


class Cutmix(Transform):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(alpha, 1) if alpha > 0 else None

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, expression):

        length = seq.shape[2]

        p = self.p_dist.sample()
        width = int(length * p / 2)

        pos = random.randint(0, length)
        x0, x1 = max(0, pos - width), min(length, pos + width)

        r_seq, r_expr = seq.roll(1, 0), expression.roll(1, 0)
        seq[:, :, x0:x1] = r_seq[:, :, x0:x1]
        expression = p * expression + (1 - p) * r_expr

        return seq, expression


class RandomErase(Transform):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.p_dist = Beta(alpha, 1) if alpha > 0 else None

    def __bool__(self):
        return self.alpha != 0

    def apply(self, seq, expression):

        length = seq.shape[2]

        p = self.p_dist.sample()
        width = int(length * p / 2)

        pos = random.randint(0, length)
        x0, x1 = max(0, pos - width), min(length, pos + width)

        seq[:, :, x0:x1] = torch.rand(seq[:, :, x0:x1].shape)

        return seq, expression


class Mutate(Transform):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __bool__(self):
        return self.n > 0

    def apply(self, seq, expression):

        length = seq.shape[2]

        pos = torch.randint(high=length, size=(self.n,))
        perm = torch.vstack([torch.randperm(4) for _ in range(self.n)]).T
        seq[:, :, pos] = seq[:, perm, pos]

        return seq, expression

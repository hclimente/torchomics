import random

import torch


def mixup(seq, rc, expression, p_dist, dataset):

    p = p_dist.sample()

    r_seq, r_rc, r_expr = item_sampler(dataset)
    seq = p * seq + (1 - p) * r_seq
    rc = p * rc + (1 - p) * r_rc
    expression = p * expression + (1 - p) * r_expr

    return seq, rc, expression


def cutmix(seq, rc, expression, p_dist, dataset):

    length = seq.shape[1]

    p = p_dist.sample()
    width = int(length * torch.sqrt(1 - p) / 2)

    pos = random.randint(0, length)
    x0, x1 = max(0, pos - width), min(length, pos + width)

    r_seq, r_rc, r_expr = item_sampler(dataset)
    seq[:, x0:x1] = r_seq[:, x0:x1]
    rc[:, x0:x1] = r_rc[:, x0:x1]
    expression = p * expression + (1 - p) * r_expr

    return seq, rc, expression


def rand_erase(seq, rc, expression, p_dist):

    length = seq.shape[1]

    p = p_dist.sample()
    width = int(length * torch.sqrt(1 - p) / 2)

    pos = random.randint(0, length)
    x0, x1 = max(0, pos - width), min(length, pos + width)

    seq[:, x0:x1] = torch.rand(seq[:, x0:x1].shape)
    rc[:, x0:x1] = torch.rand(rc[:, x0:x1].shape)

    return seq, rc, expression


def mutate(seq, rc, expression, n):

    # FIXME: this function is probably wrong
    length = seq.shape[1]

    pos = torch.randint(high=length, size=(n,))
    perm = torch.vstack([torch.randperm(4) for _ in range(n)]).T
    seq[:, :, pos] = seq[:, perm, pos]

    return seq, rc, expression


def item_sampler(dataset):
    i = random.randint(0, len(dataset) - 1)

    return dataset.sequences[i], dataset.rc_sequences[i], dataset.expression[i]

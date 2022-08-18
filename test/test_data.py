import torch
from genomic_benchmarks.dataset_getters.pytorch_datasets import (
    DemoMouseEnhancers,
    HumanEnhancersCohn,
)

from torchomics.data import import_genomics_benchmark
from torchomics.utils import one_hot_encode


def test_import_genomics_benchmark():

    dset_1 = HumanEnhancersCohn("train", version=0)
    dset_2 = DemoMouseEnhancers("train", version=0)

    for ds in [dset_1, dset_2]:
        ods = import_genomics_benchmark(ds)

        assert len(ods) == len(ds)
        assert len(ods[185]) == 2

        length = ods[185][0].shape[1]

        for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            assert torch.all(ods[i][0] == one_hot_encode(ds[i][0], padding=length))
            assert ods[i][1] == ds[i][1]

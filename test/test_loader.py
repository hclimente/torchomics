from data import DreamDM


def test_datamodule():
    dm = DreamDM(data_dir="data/dream/", accelerator="cpu")
    dm.setup()

    assert len(dm.train) == 109800
    assert len(dm.val) == 100
    assert len(dm.test) == 100
    assert len(dm.pred) == 71103

"""Config loader tests."""

from pathlib import Path

from mirt_dkvmn.config.loader import load_config


def test_load_config_base(tmp_path):
    config_path = Path("mirt-dkvmn/configs/base.yaml")
    config = load_config(config_path)

    assert config.model.n_questions > 0
    assert config.training.epochs > 0
    assert config.data.dataset_name

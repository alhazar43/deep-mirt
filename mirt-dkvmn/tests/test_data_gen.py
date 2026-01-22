"""Synthetic data generator tests."""

from pathlib import Path

from mirt_dkvmn.utils.data_gen import MirtGpcmGenerator, SyntheticConfig


def test_data_gen_writes_files(tmp_path):
    config = SyntheticConfig(
        n_students=12,
        n_questions=8,
        n_cats=4,
        n_traits=2,
        seq_len_range=(3, 6),
        seed=123,
    )
    generator = MirtGpcmGenerator(config)
    out_dir = generator.generate_and_save(str(tmp_path), "synthetic_12_8_4", split_ratio=0.75, val_ratio=0.1)

    assert (out_dir / "synthetic_12_8_4_train.txt").exists()
    assert (out_dir / "synthetic_12_8_4_valid.txt").exists()
    assert (out_dir / "synthetic_12_8_4_test.txt").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "true_irt_parameters.json").exists()

"""Data loader tests."""

import json

from mirt_dkvmn.data.loaders import DataLoaderManager


def test_loader_reads_sequences(tmp_path):
    dataset_dir = tmp_path / "demo"
    dataset_dir.mkdir()

    sequences = {
        "questions": [[1, 2, 3]],
        "responses": [[0, 1, 2]],
        "n_questions": 4,
        "n_cats": 3,
    }
    (dataset_dir / "sequences.json").write_text(json.dumps(sequences), encoding="utf-8")

    loader = DataLoaderManager("demo", data_root=str(tmp_path))
    bundle = loader.load()

    assert bundle.n_questions == 4
    assert bundle.n_cats == 3




def test_loader_reads_text_format(tmp_path):
    dataset_dir = tmp_path / "assist2015_dkvmn"
    dataset_dir.mkdir()

    text = "\n".join([
        "3",
        "1,2,3",
        "0,1,0",
        "2",
        "2,2",
        "1,1",
    ])
    (dataset_dir / "assist2015_dkvmn_train.txt").write_text(text, encoding="utf-8")

    loader = DataLoaderManager("assist2015_dkvmn", data_root=str(tmp_path))
    bundle = loader.load()

    assert bundle.n_questions == 3
    assert bundle.n_cats == 2
    assert len(bundle.questions) == 2


def test_load_splits_with_text_files(tmp_path):
    dataset_dir = tmp_path / "synthetic_4_4_3"
    dataset_dir.mkdir()

    train = "\n".join(["2", "1,2", "0,1", "2", "2,3", "1,2"])
    valid = "\n".join(["2", "1,1", "0,0"])
    test = "\n".join(["2", "2,2", "1,1"])

    (dataset_dir / "synthetic_4_4_3_train.txt").write_text(train, encoding="utf-8")
    (dataset_dir / "synthetic_4_4_3_valid.txt").write_text(valid, encoding="utf-8")
    (dataset_dir / "synthetic_4_4_3_test.txt").write_text(test, encoding="utf-8")

    loader = DataLoaderManager("synthetic_4_4_3", data_root=str(tmp_path))
    splits = loader.load_splits()

    assert len(splits["train"].questions) == 2
    assert len(splits["valid"].questions) == 1
    assert len(splits["test"].questions) == 1

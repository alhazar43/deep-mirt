"""Evaluation entry point (placeholder)."""

import argparse

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    logger = get_logger("evaluate")
    config = load_config(args.config)
    logger.info("Loaded config for dataset %s", config.data.dataset_name)


if __name__ == "__main__":
    main()

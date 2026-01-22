"""Synthetic MIRT-GPCM dataset generator."""

import argparse

from mirt_dkvmn.utils.data_gen import MirtGpcmGenerator, build_default_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="synthetic_<students>_<questions>_<cats>")
    parser.add_argument("--n_traits", type=int, default=3)
    parser.add_argument("--min_seq", type=int, default=10)
    parser.add_argument("--max_seq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    config = build_default_config(args.name, args.n_traits, args.min_seq, args.max_seq, args.seed)
    generator = MirtGpcmGenerator(config)
    output = generator.generate_and_save(args.output_dir, args.name, val_ratio=args.val_ratio)
    print(f"Wrote dataset to {output}")


if __name__ == "__main__":
    main()

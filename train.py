"""Weather Recognition — Training Entry Point.

Usage:
    python train.py                    # run single fold (FOLD_TO_RUN from config)
    python train.py --fold 2           # run specific fold
    python train.py --all              # run all k folds
"""
import argparse
import sys

from config import FOLD_TO_RUN, K_FOLDS
from utils.trainer import run_train


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fold", type=int, default=None, help="Fold index to run")
    group.add_argument("--all", action="store_true", help="Run all folds")
    args = parser.parse_args()

    if args.all:
        sys.modules["config"].FOLD_TO_RUN = -1
    elif args.fold is not None:
        sys.modules["config"].FOLD_TO_RUN = args.fold

    print(f"Running fold(s): {'all' if FOLD_TO_RUN == -1 else FOLD_TO_RUN}")
    run_train()


if __name__ == "__main__":
    main()

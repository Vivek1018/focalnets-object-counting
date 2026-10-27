from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _cmd_pretrain(args: argparse.Namespace) -> int:
    from focalnet_counting.train import pretrain

    pretrain(args.data_dir, args.epochs, args.batch_size, args.checkpoint_dir)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from focalnet_counting.train import train_counting

    train_counting(
        args.data_dir,
        args.epochs,
        args.batch_size,
        args.checkpoint_dir,
        args.encoder_checkpoint,
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from focalnet_counting.train import evaluate

    evaluate(args.data_dir, args.split, args.checkpoint)
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from focalnet_counting.model import CountingModel, ReconstructionModel

    image = torch.rand(1, 3, 128, 128)
    recon = ReconstructionModel(dim=64, depth=2, patch_size=16)
    count = CountingModel(dim=64, depth=2, patch_size=16)
    reconstructed = recon(image)
    density = count(image)
    print("reconstruction", tuple(reconstructed.shape))
    print("density", tuple(density.shape), "count", float(count.predicted_count(image)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FocalNets object counting on FSC-147."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pretrain", help="Stage 1: self-supervised reconstruction.")
    pre.add_argument("--data-dir", type=Path, default=Path("data/FSC147"))
    pre.add_argument("--epochs", type=int, default=10)
    pre.add_argument("--batch-size", type=int, default=8)
    pre.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    pre.set_defaults(func=_cmd_pretrain)

    train_p = sub.add_parser("train", help="Stage 2: density-map counting.")
    train_p.add_argument("--data-dir", type=Path, default=Path("data/FSC147"))
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    train_p.add_argument("--encoder-checkpoint", type=Path, default=None)
    train_p.set_defaults(func=_cmd_train)

    eval_p = sub.add_parser("eval", help="MAE / RMSE on a dataset split.")
    eval_p.add_argument("--data-dir", type=Path, default=Path("data/FSC147"))
    eval_p.add_argument("--split", choices=["train", "val", "test"], default="val")
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.set_defaults(func=_cmd_eval)

    demo = sub.add_parser("demo", help="Forward-pass smoke test without FSC-147.")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

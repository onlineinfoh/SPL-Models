#!/usr/bin/env python3
"""
U-Net baseline retrained with a genuine held-out validation set.

The defect
----------
The published U-Net run used `--validation 0`. In Pytorch-UNet's train.py that
takes the branch at lines 63-68:

    else:
        # use entire dataset for both training and validation metrics
        n_train = len(dataset); n_val = n_train
        train_set, val_set = dataset, dataset

so every "validation" metric was computed on the training set. The deposited log
is explicit about it: its header column is `val_dice(train)` and the values are
identical to `train_dice` (epoch 1: 0.0008 and 0.0008). The checkpoint was
therefore selected on training performance.

The fix
-------
Supply internal_val (n=257, Center 1) as a separate dataset rather than
splitting the training cohort. `--validation 0.1` would NOT be correct: it
carves a random tenth out of training, which is neither the internal_val cohort
nor the same validation set nnU-Net now uses. Using internal_val makes all three
segmentation arms share one validation set for the first time.

Everything else is held at the published values: 350 epochs, batch size 1,
learning rate 5e-4, RMSprop, ReduceLROnPlateau on Dice, gradient clipping 1.0,
n_channels=1, n_classes=2.

This driver lives in the tracked part of the repository and imports the model,
dataset and metric code from the upstream tree, which is not vendored. The
upstream tree is left unmodified.

Note on resolution: Pytorch-UNet's train.py exposes no --target-size argument,
so the published run trained at native resolution. README.md line 95 states
"512 x 512 input" for this arm, which the deposited command does not support.
Default here is native, matching what was actually run; pass --target-size to
override.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "seg-model-training" / "Pytorch-UNet"
sys.path.insert(0, str(UPSTREAM))

from unet import UNet                                      # noqa: E402
from evaluate import evaluate                              # noqa: E402
from utils.data_loading import NiftiSliceDataset           # noqa: E402
from utils.dice_score import dice_loss                     # noqa: E402

TRAIN_IMG = REPO / "data" / "train" / "imagesTr"
TRAIN_MSK = REPO / "data" / "train" / "labelsTr"
VAL_IMG = REPO / "data" / "val" / "img_v"
VAL_MSK = REPO / "data" / "val" / "seg_v"

OUT = REPO / "seg-model-training" / "unet_locked"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--momentum", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-patience", type=int, default=5,
                    help="ReduceLROnPlateau patience, as published. The schedule "
                         "is driven by TRAINING LOSS, which is smooth, so the "
                         "published patience is appropriate. See "
                         "protocol/DEVIATIONS.md D6.")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="Floor for the schedule, so it cannot reach zero.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--train-eval-every", type=int, default=10,
                    help="Evaluate the training set every N epochs. Validation "
                         "runs every epoch because it drives checkpoint "
                         "selection and the LR schedule; training metrics are "
                         "logged for monitoring only. The upstream script "
                         "evaluated all 600 training cases every epoch, which "
                         "roughly triples epoch cost for no effect on the "
                         "result.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = NiftiSliceDataset(TRAIN_IMG, TRAIN_MSK, target_size=args.target_size)
    val_set = NiftiSliceDataset(VAL_IMG, VAL_MSK, target_size=args.target_size)
    if len(train_set) == 0 or len(val_set) == 0:
        raise SystemExit("empty dataset; check data/ paths")

    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    train_eval_loader = DataLoader(train_set, shuffle=False, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=False)
    model = model.to(memory_format=torch.channels_last, device=device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum, foreach=True)
    # Driven by TRAINING LOSS, minimised. The published run drove this schedule
    # with training Dice; both are smooth, monotone-ish signals computed on data
    # the model is fitting, so the LR dynamics match the published behaviour.
    #
    # It must NOT be driven by held-out Dice. An earlier version did that, and on
    # 257 noisy validation cases the schedule decayed to its floor within ~100
    # epochs: 239 of 350 epochs then ran at 1e-6 and the model never converged,
    # collapsing the U-Net baseline from 0.797 to 0.661 internal Dice. That
    # weakened the baseline against which nnU-Net is compared, biasing the
    # comparison toward the selected model.
    #
    # Checkpoint selection remains on held-out Dice. That separation is the point:
    # the defect being corrected is which checkpoint is KEPT, not how the learning
    # rate is scheduled.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, min_lr=args.min_lr)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = nn.CrossEntropyLoss() if args.classes > 1 else nn.BCEWithLogitsLoss()

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "train_log.txt"
    cols = ["epoch", "lr", "train_loss", "train_dice", "train_mIoU",
            "train_precision", "train_recall", "train_fpr", "val_dice",
            "val_mIoU", "val_precision", "val_recall", "val_fpr"]

    logging.info(f"train {len(train_set)} slices / val {len(val_set)} slices "
                 f"(internal_val, Center 1, genuinely held out)")

    best, best_epoch = -1.0, None
    with log_path.open("w") as lf:
        lf.write("# U-Net baseline, validation on internal_val (n=257).\n")
        lf.write("# Supersedes the published run, which used --validation 0 and\n")
        lf.write("# therefore computed validation metrics on the training set.\n")
        lf.write("\t".join(cols) + "\n")

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for batch in train_loader:
                images = batch["image"].to(device=device, dtype=torch.float32,
                                           memory_format=torch.channels_last)
                true_masks = batch["mask"].to(device=device, dtype=torch.long)

                with torch.autocast(device.type, enabled=args.amp):
                    pred = model(images)
                    if args.classes == 1:
                        loss = criterion(pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(pred.squeeze(1)),
                                          true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(pred, true_masks)
                        probs = F.softmax(pred, dim=1).float()
                        onehot = F.one_hot(true_masks, args.classes).permute(0, 3, 1, 2).float()
                        # foreground only, matching the reported metric definition
                        loss += dice_loss(probs[:, 1:], onehot[:, 1:], multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += float(loss.detach()); n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            va = evaluate(model, val_loader, device, args.amp)
            # Schedule on training loss; select the checkpoint on held-out Dice.
            scheduler.step(mean_loss)

            # Training metrics are monitoring output only; nothing selects on
            # them. Evaluating 600 full-resolution cases every epoch costs far
            # more than the validation pass it accompanies.
            want_train = (epoch % args.train_eval_every == 0
                          or epoch in (1, args.epochs))
            tr = evaluate(model, train_eval_loader, device, args.amp) if want_train else None

            lr_now = optimizer.param_groups[0]["lr"]
            tr_cols = ([f"{tr[k]:.4f}" for k in ("Dice", "mIoU", "Precision", "Recall", "FPR")]
                       if tr is not None else ["", "", "", "", ""])
            lf.write("\t".join([str(epoch), f"{lr_now:.8f}", f"{mean_loss:.5f}"] + tr_cols + [
                f"{va[k]:.4f}" for k in ("Dice", "mIoU", "Precision", "Recall", "FPR")]) + "\n")
            lf.flush()

            # Checkpoint on HELD-OUT Dice. This is the line the published run
            # could not execute honestly, because its val set was the train set.
            if va["Dice"] > best + 1e-6:
                best, best_epoch = va["Dice"], epoch
                sd = model.state_dict()
                sd["mask_values"] = getattr(train_set, "mask_values", [0, 1])
                torch.save(sd, OUT / "checkpoint_best.pth")

            if epoch % 10 == 0 or epoch == 1:
                td = f"train_dice={tr['Dice']:.4f} " if tr is not None else ""
                logging.info(f"epoch {epoch}/{args.epochs} {td}"
                             f"val_dice={va['Dice']:.4f} "
                             f"(best {best:.4f} @ {best_epoch})")

        lf.write(f"# BEST epoch={best_epoch} val_dice={best:.4f}\n")

    logging.info(f"done. best internal_val Dice {best:.4f} at epoch {best_epoch}")
    logging.info(f"checkpoint: {(OUT / 'checkpoint_best.pth').relative_to(REPO)}")


if __name__ == "__main__":
    main()

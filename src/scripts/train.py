# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import imagesize
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.model import MeanScaleHyperprior

logger = logging.getLogger(__name__)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_type="all"):
        super().__init__()
        self.metric = nn.MSELoss()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            bits / num_pixels for bits in output["estimated_bits"].values()
        )

        out["mse_loss"] = self.metric(output["x_hat"], target)
        distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in different directories named under "train" and "val".:

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
        patchsize (int): size of the image patch
    """

    def __init__(self, root, split="train", patchsize=256):
        splitdir = Path(root) / f"{split}"
        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samples = [
            f
            for f in self.samples
            if imagesize.get(f)[0] >= patchsize and imagesize.get(f)[1] >= patchsize
        ]

        if split == "train":
            self.transform = transforms.Compose(
                (
                    transforms.ToImage(),
                    transforms.RandomCrop(patchsize),
                    transforms.ToDtype(torch.float32, scale=True),
                )
            )
        else:
            self.transform = transforms.Compose(
                (
                    transforms.ToImage(),
                    transforms.CenterCrop(patchsize),
                    transforms.ToDtype(torch.float32, scale=True),
                )
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def make_infinite(dloader):
    while True:
        yield from dloader


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    epoch,
    epoch_size=1000,
    clip_max_norm=1.0,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(make_infinite(train_dataloader)):
        if i > epoch_size:
            break
        d = d.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i} / {epoch_size}"
                f" ({100.0 * i / epoch_size:.0f}%)]"
                f"\tLoss: {out_criterion['loss'].item():.3f} |"
                f"\tMSE loss: {out_criterion['mse_loss'].item():.3f} |"
                f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tPSNR: {10 * math.log10(1 / mse_loss.avg):.2f} |"
    )

    return loss.avg


def save_checkpoint(state, is_best, savepath, filename="checkpoint.pth.tar"):
    os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath, os.path.join(savepath, "checkpoint_best_loss.pth.tar")
        )


def main(args):
    os.makedirs(args.logpath)
    logging.basicConfig(
        filename=os.path.join(args.logpath, f"train_{args.lmbda}.log"),
        level=logging.INFO,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = ImageFolder(args.datapath, split="train")
    test_dataset = ImageFolder(args.datapath, split="valid")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = MeanScaleHyperprior(128, 192)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.epochsize,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                os.path.join(args.savepath, str(args.lmbda)),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patchsize", default=256, type=int)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--datapath", default="./datasets", type=str)
    parser.add_argument("--savepath", default="./save", type=str)
    parser.add_argument("--lmbda", default=0.013, type=float)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--epochsize", default=1000, type=int)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--logpath", default="logs", type=str)
    args = parser.parse_args()
    main(args)

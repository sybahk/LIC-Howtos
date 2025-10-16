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
from pathlib import Path

import imagesize
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patchsize", default=256, type=int)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--datapath", default="./dataset", type=str)
    parser.add_argument("--savepath", default="./save", type=str)
    parser.add_argument("--lmbda", default=0.013, type=float)
    args = parser.parse_args()

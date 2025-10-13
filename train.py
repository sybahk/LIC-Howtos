import argparse

import numpy as np
import tensorflow_datasets as tfds
import torch
import torchvision.transforms.v2 as transforms


def check_image_size(image, patchsize):
    shape = np.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def get_dataset(name, split, args):
    """Creates input data pipeline from a TF Datasets dataset."""
    dataset = tfds.as_numpy(tfds.load(name, split=split, shuffle_files=True))
    if split == "train":
        dataset = dataset.repeat()
        transform = transforms.Compose(
            (
                transforms.ToImage(),
                transforms.RandomCrop(args.patchsize),
                transforms.ToDtype(torch.float32, scale=True),
            )
        )
    else:
        transform = transforms.Compose(
            (
                transforms.ToImage(),
                transforms.CenterCrop(args.patchsize),
                transforms.ToDtype(torch.float32, scale=True),
            )
        )
    dataset = dataset.filter(lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(lambda x: transform(x))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patchsize", default=256, type=int)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--savepath", default="./save", type=str)
    parser.add_argument("--lmbda", default=0.013, type=float)
    parser.parse_args()

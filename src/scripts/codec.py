import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.ops as op
from src.model import MeanScaleHyperprior

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CodecInfo(NamedTuple):
    original_size: tuple
    original_bitdepth: int
    net: nn.Module
    device: str


def encode_image(input, codec: CodecInfo, output):
    img = op.load_image(input)
    x = op.img2torch(img).to(codec.device)
    bitdepth = 8

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = op.pad(x, p)

    with torch.no_grad():
        out = codec.net.compress(x)

    shape = out["shape"]

    with Path(output).open("wb") as f:
        # write original image size
        op.write_uints(f, (h, w))
        # write original bitdepth
        op.write_uchars(f, (bitdepth,))
        # write shape and number of encoded latents
        op.write_body(f, shape, out["strings"])

    size = op.filesize(output)
    bpp = float(size) * 8 / (h * w)

    return {"bpp": bpp}


@torch.no_grad
def encode(args):
    net = MeanScaleHyperprior(128, 192)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    net = net.to(device=device)
    net = net.eval()
    state_dict = torch.load(args.modelpath)
    lmbda = state_dict["lmbda"]
    net.load_state_dict(state_dict["state_dict"])
    os.makedirs(os.path.join(args.bitstreampath, str(lmbda)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.bitstreampath, str(lmbda), f"encode_{lmbda}.log"), "w"
            ),
            logging.StreamHandler(),
        ],
    )
    for f in os.scandir(args.inputpath):
        if f.is_file() and f.name.endswith(".png"):
            img = op.load_image(f)
            x = op.img2torch(img).to(device)
            bitdepth = 8
            h, w = x.size(2), x.size(3)
            p = 64  # maximum 6 strides of 2
            x = op.pad(x, p)

            with torch.no_grad():
                out = net(x, include_strings=True)
            basename = f.name.split(".")[0]
            outputpath = os.path.join(args.bitstreampath, str(lmbda), basename + ".bin")
            shape = out["compressed"]["shape"]

            with open(outputpath, "wb") as f:
                # write original image size
                op.write_uints(f, (h, w))
                # write original bitdepth
                op.write_uchars(f, (bitdepth,))
                # write shape and number of encoded latents
                op.write_body(f, shape, out["compressed"]["strings"])

            size = op.filesize(outputpath)
            bpp = float(size) * 8 / (h * w)
            estimated_bpp = (
                out["estimated_bits"]["y"] + out["estimated_bits"]["z"]
            ) / (h * w)
            psnr = 10 * math.log10(1 / F.mse_loss(x, out["x_hat"]))
            logger.info(
                f"Encoded {basename}: actual bpp: {bpp:.4f}, estimated_bpp: {estimated_bpp:.4f}, psnr: {psnr:.2f}dB",
            )


def decode_image(f, codec: CodecInfo, output):
    strings, shape = op.read_body(f)
    with torch.no_grad():
        out = codec.net.decompress(strings, shape)

    x_hat = op.crop(out["x_hat"], codec.original_size)

    img = op.torch2img(x_hat)

    if output is not None:
        img.save(output)
    return {"img": img}


@torch.no_grad
def decode(args):
    net = MeanScaleHyperprior(128, 192)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    net = net.to(device=device)
    net = net.eval()
    state_dict = torch.load(args.modelpath)
    lmbda = state_dict["lmbda"]
    net.load_state_dict(state_dict["state_dict"])
    os.makedirs(os.path.join(args.imagepath, str(lmbda)), exist_ok=True)

    for file in os.scandir(os.path.join(args.inputpath, str(lmbda))):
        if file.is_file() and file.name.endswith(".bin"):
            with open(file, "rb") as f:
                original_size = op.read_uints(f, 2)
                original_bitdepth = op.read_uchars(f, 1)[0]
                outputpath = os.path.join(
                    args.imagepath, str(lmbda), file.name.split(".")[0] + ".png"
                )
                decode_image(
                    f,
                    CodecInfo(original_size, original_bitdepth, net, device),
                    outputpath,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Encode or Decode images using trained models.")
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "encode":
        encoder_parser = argparse.ArgumentParser("Encode images into bitstream")
        encoder_parser.add_argument(
            "--inputpath",
            default="datasets/test",
            type=str,
            help="Path to the input images.",
        )
        encoder_parser.add_argument(
            "--modelpath",
            type=str,
            required=True,
            help="Path to the weight of the trained model.",
        )
        encoder_parser.add_argument(
            "--bitstreampath",
            type=str,
            default="outputs/bitstreams",
            help="Path to write bitstreams of images.",
        )
        encoder_parser.add_argument("--cuda", action="store_true")
        encoder_args = encoder_parser.parse_args(sys.argv[2:])
        encode(encoder_args)

    elif args.command == "decode":
        decoder_parser = argparse.ArgumentParser()
        decoder_parser = argparse.ArgumentParser("Decode images into bitstream")
        decoder_parser.add_argument(
            "--inputpath",
            type=str,
            default="outputs/bitstreams",
            help="Path to the bitstreams of images.",
        )
        decoder_parser.add_argument(
            "--modelpath",
            type=str,
            required=True,
            help="Path to the weight of the trained model.",
        )
        decoder_parser.add_argument(
            "--imagepath",
            type=str,
            default="outputs/images",
            help="Path to write decoded images.",
        )
        decoder_parser.add_argument("--cuda", action="store_true")
        decoder_args = decoder_parser.parse_args(sys.argv[2:])
        decode(decoder_args)

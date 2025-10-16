import unittest

import torch

from src.model import MeanScaleHyperprior

torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class TestMeanScaleHyperprior(unittest.TestCase):
    @torch.no_grad
    def test_cvc_equal_between_results(self):
        net = MeanScaleHyperprior(64, 96).cuda()
        net.eval()

        x = torch.empty(1, 3, 128, 128).uniform_().cuda()
        out = net(x)
        out_forwarded = out["x_hat"].clamp_(0, 1)
        out_compressed = net(x, include_strings=True)["compressed"]
        out_decompressed = net.decompress(**out_compressed)["x_hat"].clamp_(0, 1)
        print(torch.max(abs(out_forwarded - out_decompressed)))
        self.assertEqual(out_forwarded.shape, (1, 3, 128, 128))
        self.assertEqual(out_decompressed.shape, (1, 3, 128, 128))
        self.assertTrue(torch.allclose(out_forwarded, out_decompressed))


if __name__ == "__main__":
    unittest.main()

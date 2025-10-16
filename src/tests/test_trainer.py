import unittest
from dataclasses import dataclass

from src.scripts.train import get_dataset


@dataclass
class FakeArguments:
    patch_size: int = 256
    batch_size: int = 8


class TestTrainer(unittest.TestCase):
    def test_dataset(self):
        args = FakeArguments()
        train_dataset = get_dataset("clic", "train", args)
        validation_dataset = get_dataset("clic", "validation", args)
        print(next(train_dataset).shape)
        print(next(validation_dataset).shape)


if __name__ == "__main__":
    unittest.main()

import unittest
from parameterized import parameterized
import itertools

from metrics.loader import CheckpointLoader
from metrics.loader.calculate_metrics import ArchitectureSizeMetrics


class MetricsLoaderTester(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


class MetricsCalculatorTester(unittest.TestCase):
    ALL_PRETRAINED_CHECKPOINTS_PATH = "C:\Results\output"
    ALL_FINETUNE_CHECKPOINTS_PATH = "C:\Results\output_sweep_complete"
    ALL_DATASETS_PATH = "C:\DataSets"

    def setUp(self) -> None:
        self.cp_loader = CheckpointLoader(self.ALL_PRETRAINED_CHECKPOINTS_PATH,
                                          self.ALL_FINETUNE_CHECKPOINTS_PATH,
                                          self.ALL_DATASETS_PATH)

    @parameterized.expand([list(element) for element in list(itertools.product(
        ["cifar10", "cifar100", "svhn", "tinyimagenet"],
        ["lowres_resnet9", "lowres_resnet50", "lowres_densenet121", "lowres_vgg16_bn"],
        [
            (0, "cifar10_model"),
            (1, "cifar100_model"),
            (3, "svhn_model"),
            (4, "tinyimagenet_model")
        ]
    ))])
    def test_architecture_size(self, dataset, model, checkpoint):
        metrics_dict = ArchitectureSizeMetrics().calculate_metrics(
            self.cp_loader.load_pretrained_checkpoint(dataset, model, checkpoint))


if __name__ == '__main__':
    unittest.main()

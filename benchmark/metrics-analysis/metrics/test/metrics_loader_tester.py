import os
import unittest

import pandas as pd
from parameterized import parameterized
import itertools

from metrics import utils
from metrics.loader import CheckpointLoader
from metrics.loader.calculate_metrics import ArchitectureSizeMetrics, QualityMetrics, LatentSpaceMetrics


class MetricsLoaderTester(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


class MetricsCalculatorTester(unittest.TestCase):
    ALL_PRETRAINED_CHECKPOINTS_PATH = "C:\\Results\\results\\output"
    ALL_FINETUNE_CHECKPOINTS_PATH = "C:\\Results\\results\\output_sweep_complete"
    ALL_DATASETS_PATH = "C:\\DataSets"

    ARCHITECTURE_SIZE_COMPARE_PATH = "C:\\Results\\architecture_size_compare.csv"
    QUALITY_METRICS_COMPARE_PATH = "C:\\Results\\quality_compare.csv"
    LATENT_SPACE_METRICS_COMPARE_PATH = "C:\\Results\\latent_space_compare.csv"

    def setUp(self) -> None:
        self.cp_loader = CheckpointLoader(self.ALL_PRETRAINED_CHECKPOINTS_PATH,
                                          self.ALL_FINETUNE_CHECKPOINTS_PATH,
                                          self.ALL_DATASETS_PATH)

        self.architecture_size_compare_df = pd.read_csv(self.ARCHITECTURE_SIZE_COMPARE_PATH)
        del self.architecture_size_compare_df['index']

        self.quality_compare_df = pd.read_csv(self.QUALITY_METRICS_COMPARE_PATH)
        del self.quality_compare_df['index']

        self.latent_space_compare_df = pd.read_csv(self.LATENT_SPACE_METRICS_COMPARE_PATH)
        del self.latent_space_compare_df['index']


    def get_metrics_dicts(self, metric, complete_metrics_df, dataset, model, checkpoint):
        original_dict = {}
        original_dict.update({"dataset": dataset, "model": model, "checkpoint": checkpoint[1]})
        metrics_dict = metric.calculate_metrics(
            self.cp_loader.load_pretrained_checkpoint(dataset, model, (0, "Standard")))
        original_dict.update(metrics_dict)

        original_df = pd.DataFrame.from_records([original_dict])

        df = original_df[
            (original_df['dataset'] == dataset) &
            (original_df['model'] == model) &
            (original_df['checkpoint'] == checkpoint[1])
            ].dropna(axis=1).convert_dtypes("float64").reset_index(drop=True)

        compare_df = complete_metrics_df[
            (complete_metrics_df['dataset'] == dataset) &
            (complete_metrics_df['model'] == model) &
            (complete_metrics_df['checkpoint'] == checkpoint[1])
            ].dropna(axis=1).convert_dtypes("float64").reset_index(drop=True)

        return df.to_dict(), compare_df.to_dict()

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

        architecture_size_dict, architecture_size_dict_compare = self.get_metrics_dicts(
            ArchitectureSizeMetrics(),
            self.architecture_size_compare_df,
            dataset,
            model,
            checkpoint
        )

        self.assertEqual(architecture_size_dict, architecture_size_dict_compare)

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
    def test_quality_metrics(self, dataset, model, checkpoint):
        architecture_size_dict, architecture_size_dict_compare = self.get_metrics_dicts(
            QualityMetrics(),
            self.quality_compare_df,
            dataset,
            model,
            checkpoint
        )
        self.assertTrue(utils.dicts_almost_equal(architecture_size_dict, architecture_size_dict_compare))

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
    def test_latent_space_metrics(self, dataset, model, checkpoint):
        architecture_size_dict, architecture_size_dict_compare = self.get_metrics_dicts(
            LatentSpaceMetrics('kmeans', random_state=0),
            self.latent_space_compare_df,
            dataset,
            model,
            checkpoint
        )
        print(architecture_size_dict)
        print(architecture_size_dict_compare)
        # self.assertTrue(utils.dicts_almost_equal(architecture_size_dict, architecture_size_dict_compare), 0.1)

    def create_metrics(self, dataset, model, checkpoint, path):
        metrics_dict = LatentSpaceMetrics('kmeans', random_state=0).calculate_metrics(
            self.cp_loader.load_pretrained_checkpoint(dataset, model, (0, "Standard")))
        if os.path.exists(path):
            pass
            loaded_rows = pd.read_csv(path, index_col="index")
            appendix = {}
            appendix.update({"dataset": dataset, "model": model, "checkpoint": checkpoint[1]})
            appendix.update(metrics_dict)
            df = pd.concat([loaded_rows, pd.DataFrame.from_records([appendix])], axis=0)
        else:
            appendix = {}
            appendix.update({"dataset": dataset, "model": model, "checkpoint": checkpoint[1]})
            appendix.update(metrics_dict)
            print(appendix)
            df = pd.DataFrame.from_records([appendix])

        df.to_csv(path, index_label="index")


if __name__ == '__main__':
    unittest.main()

import os

import pandas as pd

from metrics.loader.calculate_metrics import Metrics, QualityMetrics, ArchitectureSizeMetrics, LatentSpaceMetrics
from metrics.loader.checkpoint_loader import CheckpointLoader


class MetricsLoader:
    """
    Loads all the given metrics into a pandas dataframe for further processing
    """

    def __init__(
            self,
            pretrained_cp_loader: CheckpointLoader,
            pretrained_checkpoint: tuple,
            finetune_cp_loader: CheckpointLoader,
            finetune_checkpoints: list[tuple],
            from_scrape_cp_loader: CheckpointLoader,
            from_scratch_checkpoint: tuple,
            finetune_datasets: list[tuple],
            pretrained_datasets: list[tuple],
            models: list,
            metrics: list[Metrics],
            use_finetune_dataset=False,

    ):
        """
        Initialize the metrics loader
        :param pretrained_cp_loader: checkpoint loader for the pretrained models
        :param pretrained_checkpoint: checkpoint on which the models were pretrained
        :param finetune_cp_loader:  checkpoint loader for the finetuned models
        :param finetune_checkpoints: checkpoints on which were finetuned on
        :param from_scrape_cp_loader: checkpoint loader for the from-scrape models
        :param from_scratch_checkpoint: checkpoint on which were from-scrape training was performed
        :param finetune_datasets: list of datasets on which the metrics are applied on
        :param models: list of models on which the metrics are applied on
        :param metrics: Metrics to be accumulated in the final dataframe
        :param use_finetune_dataset: Set if the metrics use the finetune dataset
        """
        self.pretrained_cp_loader = pretrained_cp_loader
        self.pretrained_checkpoint = pretrained_checkpoint
        self.finetune_cp_loader = finetune_cp_loader
        self.finetune_checkpoints = finetune_checkpoints
        self.from_scrape_cp_loader = from_scrape_cp_loader
        self.from_scratch_checkpoint = from_scratch_checkpoint
        self.finetune_datasets = finetune_datasets
        self.pretrained_datasets = pretrained_datasets
        self.models = models
        self.metrics = metrics
        self.use_finetune_dataset = use_finetune_dataset

    def __get_all_metrics(self, model, config=None):
        """
        Calculate and concat all metrics for the given model
        :param model: model on which the metrics are calculated on
        :return: all metrics combined in a dictionary
        """
        all_metrics = {}
        for metric in self.metrics:
            if config is not None or config:
                metric.update_config(config)
            all_metrics.update(metric.calculate_metrics(model))
        return all_metrics

    def __get_difference(self, model, finetune_dataset, finetune_checkpoint, from_scratch_checkpoint) -> dict:
        """
        Calculates the difference in accuracy between the given pretrained checkpoint and finetune checkpoint
        :param model: model trained on
        :param finetune_dataset: dataset finetuned on
        :param finetune_checkpoint: finetune checkpoint to calculate difference
        :param from_scratch_checkpoint: from-scratch-checkpoint from the simultaneous finetune checkpoint
        :return: the difference between the accuracy of the finetune and from-scratch checkpoint
        """
        df_tmp = pd.read_csv(
            os.path.join(self.finetune_cp_loader.all_checkpoints_path, finetune_dataset[0], model,
                         f"version_{finetune_checkpoint[0]}",
                         "metrics.csv"))
        acc = float(df_tmp[df_tmp['acc/val'] == df_tmp['acc/val'].max()]['acc/val'].values[0])

        df_compare_tmp = pd.read_csv(
            os.path.join(self.from_scrape_cp_loader.all_checkpoints_path, finetune_dataset[0], model,
                         f"version_{from_scratch_checkpoint[0]}",
                         "metrics.csv"))
        acc_compare = float(
            df_compare_tmp[df_compare_tmp['acc/val'] == df_compare_tmp['acc/val'].max()]['acc/val'].values[0])
        return {"difference": acc - acc_compare}

    def load_all_metrics(self) -> pd.DataFrame:
        """
        Loads all the metrics from the class given value range of models, datasets and checkpoints
        :return: DataFrame with all metrics
        """
        records = []
        for pretrained_dataset in self.pretrained_datasets:
            for model in self.models:
                for finetune_checkpoint, finetune_dataset in zip(self.finetune_checkpoints, self.finetune_datasets):

                    config = {}
                    if self.use_finetune_dataset:
                        config['finetune_dataset'] = finetune_dataset

                        all_metrics = self.__get_all_metrics(
                            model=self.pretrained_cp_loader.load_checkpoint(finetune_dataset, model,
                                                                            self.pretrained_checkpoint),
                            config=config)

                    else:
                        all_metrics = self.__get_all_metrics(
                            model=self.pretrained_cp_loader.load_checkpoint(pretrained_dataset, model,
                                                                            self.pretrained_checkpoint))

                    difference = self.__get_difference(model, finetune_dataset, finetune_checkpoint,
                                                       self.from_scratch_checkpoint)

                    record_list = {
                        "finetune_dataset": f"{finetune_dataset[0]}({finetune_dataset[1]})",
                        "pretrained_dataset": f"{pretrained_dataset[0]}({pretrained_dataset[1]})",
                        "model": model,
                        "checkpoint": finetune_checkpoint[1]
                    }
                    record_list.update(all_metrics)
                    record_list.update(difference)

                    records.append(record_list)
        return pd.DataFrame.from_records(records)

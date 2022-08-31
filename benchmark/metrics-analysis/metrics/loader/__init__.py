import pandas as pd

from metrics.loader.calculate_metrics import Metrics
from metrics.loader.checkpoint_loader import CheckpointLoader


class MetricsLoader:
    """
    Loads all the given metrics into a pandas dataframe for further processing
    """

    def __init__(self, cp_loader: CheckpointLoader, metrics: list[Metrics], datasets: list, models: list,
                 finetune_checkpoints: list[dict], pretrained_checkpoint):
        """
        Initialize the metrics loader
        :param cp_loader: Initialized checkpoint loader to work with
        :param metrics: Metrics to be accumulated in the final dataframe
        :param datasets: list of datasets on which the metrics are applied on
        :param models: list of models on which the metrics are applied on
        :param finetune_checkpoints: checkpoints on which were finetuned on
        :param pretrained_checkpoint: checkpoint on which the models were pretrained
        """
        self.pretrained_checkpoint = pretrained_checkpoint
        self.cp_loader = cp_loader
        self.finetune_checkpoints = finetune_checkpoints
        self.models = models
        self.datasets = datasets
        self.metrics = metrics

    def __get_all_metrics(self, model):
        """
        Calculate and concat all metrics for the given model
        :param model: model on which the metrics are calculated on
        :return: all metrics combined in a dictionary
        """
        all_metrics = {}
        for metric in self.metrics:
            all_metrics.update(metric.calculate_metrics(model))
        return all_metrics

    def load_all_metrics(self) -> pd.DataFrame:
        """
        Loads all the metrics from the class given value range of models, datasets and checkpoints
        :return: DataFrame with all metrics
        """
        records = []
        for dataset in self.datasets:
            for model in self.models:
                for checkpoint in self.finetune_checkpoints:
                    all_metrics = self.__get_all_metrics(
                        self.cp_loader.load_pretrained_checkpoint(dataset, model, checkpoint))
                    difference = self.cp_loader.get_difference(model, dataset, self.pretrained_checkpoint, checkpoint)

                    record_list = {}
                    record_list.update(all_metrics)
                    record_list.update(difference)

                    records.append(record_list)
        return pd.DataFrame.from_records(records)

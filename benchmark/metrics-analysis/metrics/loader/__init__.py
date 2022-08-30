import pandas as pd

from metrics.loader.calculate_metrics import Metrics
from metrics.loader.checkpoint_loader import CheckpointLoader


class MetricsLoader:
    def __init__(self, cp_loader: CheckpointLoader, metrics: list[Metrics], datasets: list, models: list,
                 finetune_checkpoints: list[dict], pretrained_checkpoint):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.cp_loader = cp_loader
        self.finetune_checkpoints = finetune_checkpoints
        self.models = models
        self.datasets = datasets
        self.metrics = metrics

    def __get_all_metrics(self, model):
        all_metrics = {}
        for metric in self.metrics:
            all_metrics.update(metric.calculate_metrics(model))
        return all_metrics

    def load_all_metrics(self) -> pd.DataFrame:
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

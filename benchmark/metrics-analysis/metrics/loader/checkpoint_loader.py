import glob
import os

import pandas as pd
import torch
import yaml

from module import TrainModule


class CheckpointLoader:

    def __init__(self, all_pretrained_checkpoints_path, all_finetune_checkpoints_path, all_datasets_path):
        self.all_finetune_checkpoints_path = all_finetune_checkpoints_path
        self.all_pretrained_checkpoints_path = all_pretrained_checkpoints_path
        self.all_datasets_path = all_datasets_path

    def load_finetune_checkpoint(self, dataset, model, checkpoint) -> torch.nn.Module:
        return self._load_checkpoint(self.all_finetune_checkpoints_path, dataset, model, checkpoint)

    def load_pretrained_checkpoint(self, dataset, model, checkpoint):
        return self._load_checkpoint(self.all_pretrained_checkpoints_path, dataset, model, checkpoint)

    def __load_hparams(self, path):
        with open(path,
                  'r') as stream:
            data_loaded = yaml.safe_load(stream)

        hparams = data_loaded['hparams']

        hparams.update({"data_dir": self.all_datasets_path})
        hparams.update({"output_dir": "/"})

        if "start_step" not in hparams:
            hparams['start_step'] = 0
        if "dataset_percentage" not in hparams:
            hparams['dataset_percentage'] = 100

        return hparams

    def _load_checkpoint(self, checkpoint_path, dataset, model, checkpoint) -> torch.nn.Module:

        hparams = self.__load_hparams(os.path.join(checkpoint_path, dataset, model, f"version_{checkpoint[0]}",
                                                   "hparams.yaml"))

        loaded_model = TrainModule(hparams)
        # print(glob.glob(os.path.join(path,dataset,model,f"version_{checkpoint[0]}","checkpoints","epoch*.ckpt"))[0])
        state = torch.load(
            glob.glob(os.path.join(checkpoint_path, dataset, model, f"version_{checkpoint[0]}",
                                   "checkpoints",
                                   "epoch*.ckpt"))[0],
            map_location=loaded_model.device)

        if "state_dict" in state:
            state = state["state_dict"]

        loaded_model.model.load_state_dict(
            dict((key.replace("model.classifier.", "model.fc.").replace("model.", ""), value) for (key, value) in
                 state.items()))
        return loaded_model

    def get_difference(self, model, dataset, pretrained_checkpoint, finetune_checkpoint) -> dict:
        df_tmp = pd.read_csv(
            os.path.join(self.all_finetune_checkpoints_path, dataset, model, f"version_{finetune_checkpoint[0]}",
                         "metrics.csv"))
        acc = float(df_tmp[df_tmp['acc/val'] == df_tmp['acc/val'].max()]['acc/val'].values[0])

        df_compare_tmp = pd.read_csv(
            os.path.join(self.all_pretrained_checkpoints_path, dataset, model, f"version_{pretrained_checkpoint[0]}",
                         "metrics.csv"))
        acc_compare = float(
            df_compare_tmp[df_compare_tmp['acc/val'] == df_compare_tmp['acc/val'].max()]['acc/val'].values[0])
        return {"difference": acc - acc_compare}

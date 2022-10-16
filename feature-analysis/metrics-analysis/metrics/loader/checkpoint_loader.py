import glob
import os

import torch
import yaml

from module import TrainModule


class CheckpointLoader:
    """
    Loads the pretrained and finetune checkpoints from given paths to make further processing of these
    checkpoints easy.
    """

    def __init__(self, all_checkpoints_path, all_datasets_path):
        """
        Initialize the paths, where the checkpoints are located
        :param all_checkpoints_path Path of the checkpoints
        :param all_datasets_path: Path of the used datasets
        """
        self.all_checkpoints_path = all_checkpoints_path
        self.all_datasets_path = all_datasets_path

    def __load_hparams(self, dataset: tuple[str, int], model: str, checkpoint: tuple[int, str]):
        """
        Loads the hyperparameter from the given path and prepares them for the further processing
        :param dataset: tuple of the dataset name and the percentage used (name, percentage)
        :param model: name of the used model
        :param checkpoint: tuple of the checkpoint version and the name given to the checkpoint (version_number, name)
        :return: loaded hyperparameters from the given checkpoint parameters
        """

        path = os.path.join(self.all_checkpoints_path, dataset[0], model, f"version_{checkpoint[0]}", "hparams.yaml")

        with open(path,
                  'r') as stream:
            data_loaded = yaml.safe_load(stream)

        hparams = data_loaded['hparams']

        hparams.update({"data_dir": self.all_datasets_path})
        hparams.update({"output_dir": "/"})

        if "start_step" not in hparams:
            hparams['start_step'] = 0
        hparams['dataset_percentage'] = dataset[1]

        return hparams

    def load_checkpoint(self, dataset: tuple[str, int], model: str, checkpoint: tuple[int, str]) -> torch.nn.Module:
        """
        Loads checkpoint from the given checkpoint path
        :param dataset: tuple of the dataset name and the percentage used (name, percentage)
        :param model: name of the used model
        :param checkpoint: tuple of the checkpoint version and the name given to the checkpoint (version_number, name)
        :return: loaded checkpoint from the given checkpoint parameters
        """

        hparams = self.__load_hparams(dataset, model, checkpoint)

        loaded_model = TrainModule(hparams)
        state = torch.load(
            glob.glob(os.path.join(self.all_checkpoints_path, dataset[0], model, f"version_{checkpoint[0]}",
                                   "checkpoints",
                                   "epoch*.ckpt"))[0],
            map_location=loaded_model.device)

        if "state_dict" in state:
            state = state["state_dict"]

        loaded_model.model.load_state_dict(
            dict((key.replace("model.classifier.", "model.fc.").replace("model.", ""), value) for (key, value) in
                 state.items()))
        return loaded_model

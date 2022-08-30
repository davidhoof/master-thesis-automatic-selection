import os
import shutil
import glob
import wandb


class Utils:
    @staticmethod
    def get_max_str(lst: list) -> str:
        """
        Returns the longest string in list
        :param lst: List of strings
        :return: Longest string
        """
        return str(max(lst, key=len))


class SweepExtractor:
    """
    Sweep-Extractor extracts the best runs from given Sweep and copies them to the output_to_path as
    continuable result
    """

    def __init__(self, sweep_name: str, root_path: str, output_from: str, output_to_path: str, checkpoint_order: dict):
        """
        Sweep-Extractor extracts the best runs from given Sweep and copies them to the output_to_path as
        continuable result
        :param sweep_name: Name of the Sweep
        :param root_path: Path of the checkpoint root
        :param output_from: Path of the output folder dependent on the root_path
        :param output_to_path: Complete path of the folder for the final result
        :param checkpoint_order: The order in which the checkpoints are run in the sweep. Needed to assign sweeps with
        checkpoints. Is a dict in shape {checkpoint_dataset_name: order_position_as_int}
        """
        self.sweep_name = sweep_name
        self.root_path = root_path
        self.output_from = output_from
        self.output_to_path = output_to_path
        self.checkpoint_order = checkpoint_order

    def find_identifiers(self, run_name):
        """
        Finds the identifiers from the checkpoints in a specific run
        :param run_name: Name of the run
        :return: identifier tokens as (model, dataset, run)
        """
        paths = glob.glob(os.path.join(self.root_path, f"lowres_*/*{run_name}"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Run {run_name} could not be found")
        glob_tokens = paths[0].split("/")
        path_tokens = Utils.get_max_str(glob_tokens).split("_")
        run_tokens = glob_tokens[3].split(f"_{run_name}")[0]
        run = run_tokens.split(f"_{run_name}")[0]
        if len(path_tokens) == 4:
            model = f"{path_tokens[0]}_{path_tokens[1]}_{path_tokens[2]}"
            dataset = path_tokens[3].split(self.sweep_name)[1]
        else:
            model = f"{path_tokens[0]}_{path_tokens[1]}"
            dataset = path_tokens[2].split(self.sweep_name)[1]
        return model, dataset, run

    def copy_sweep(self, run_name):
        """
        Copies a single sweep to the output_path
        :param run_name: Name of the run
        :return: Run and Dataset as tuple: (run, dataset)
        """
        model, dataset, run = self.find_identifiers(run_name)
        from_path = os.path.join(self.root_path, self.output_from, dataset, model, f"version_{run}")
        to_path = os.path.join(self.output_to_path, dataset, model, f"version_{run}")
        from_path_checkpoint = glob.glob(os.path.join(self.root_path, f"lowres_*/*{run_name}/checkpoints"))[0]
        to_path_checkpoint = os.path.join(self.output_to_path, dataset, model, f"version_{run}/checkpoints")

        appendix = 2
        while os.path.exists(to_path):
            to_path = os.path.join(self.output_to_path, dataset, model, f"version_{run}_{appendix}")
            to_path_checkpoint = os.path.join(self.output_to_path, dataset, model,
                                              f"version_{run}_{appendix}/checkpoints")
            appendix = appendix + 1
        shutil.copytree(from_path, to_path)
        shutil.copytree(from_path_checkpoint, to_path_checkpoint)
        return run, dataset

    @staticmethod
    def get_best_run(sweep_id):
        """
        Gets best run from the wandb api by the sweep_id
        :param sweep_id: Specific sweep id for the wandb api
        :return: Model, Dataset(trained_on) and id of best run (model, dataset_trained_on, id)
        """
        api = wandb.Api()
        sweep = api.sweep(sweep_id)
        model, dataset, _id = sweep.best_run(order="summary_metrics.acc_max/val").name.split("-")
        dataset_trained_on = str(sweep.name).replace(model, "")
        return model, dataset_trained_on, _id

    @staticmethod
    def get_sweeps(command_path):
        """
        Gets all sweeps from the command file path. This file is created by the sweep module of the
        pytorch-pretrained-CNNs framework. The Framework is found on:
        https://github.com/davidhoof/pytorch-pretrained-cnns
        :param command_path: Path to the command list
        :return: List of all sweeps
        """
        with open(command_path, "r") as file:
            lines = file.readlines()
            sweep_list = [line.rstrip().split(" ")[2] for line in lines]
        return sweep_list

    def copy_sweeps(self, sweep_command_path):
        """
        Copies the best run from all sweeps from the command path to the output_to_path
        :param sweep_command_path:
        :return:
        """
        sweeps = SweepExtractor.get_sweeps(sweep_command_path)
        for sweep in sweeps:
            try:
                model, dataset_trained_on, best_run_id = SweepExtractor.get_best_run(sweep)
                run_id, dataset = self.copy_sweep(best_run_id)
                os.rename(os.path.join(self.output_to_path, dataset, model, f"version_{run_id}"),
                          os.path.join(self.output_to_path, dataset, model,
                                       f"version_{self.checkpoint_order[dataset_trained_on]}"))
            except:
                print("could not copy")

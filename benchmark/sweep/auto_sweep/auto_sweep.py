import glob
import multiprocessing as mp
import os
from argparse import ArgumentParser

import wandb
import yaml


def gpu_worker(args):
    """
    GPU-Worker to run Sweep on given GPU
    :param args: (gpu_id,queue) contains a tuple of the assigned gpu id and the workqueue
    :return:
    """
    gpu_id = args[0]
    q = args[1]

    while not q.empty():
        (sweep_id, number_of_runs) = q.get()
        username, project, sweep_id = sweep_id.split("/")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        wandb.agent(sweep_id, count=number_of_runs, project=project, entity=username)


def main(config_):
    """
    Executes the sweeps with the given parameters
    :param config_: Given config from the auto_sweep_config
    :return:
    """
    root_folder = config_['root']
    sweep_name = config_['sweep_name']
    datasets = config_['datasets']

    gpus = config_['gpus']
    models = config_['models']
    number_of_runs = config_['number_of_runs']

    pool = mp.Pool(len(gpus))
    m = mp.Manager()

    queues = {}
    for gpu, model in zip(gpus, models):
        queues[model] = m.Queue()

    sweep_list = []
    for dataset in datasets:
        with open(glob.glob(os.path.join(root_folder, f"{sweep_name}{dataset}", "sweep_agent_commands*.txt"))[
                      -1]) as file:
            lines = file.readlines()
            sweep_list = [(line.rstrip().split(" ")[-1], line.rstrip().split(" ")[-2]) for line in lines]

    for sweep in sweep_list:
        for model in models:
            if model == sweep[0]:
                queues[model].put((sweep[1], number_of_runs))

    que_list = list(queues.values())

    pool.map(gpu_worker, zip(gpus, que_list))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('configuration_file')

    _args = parser.parse_args()

    if type(_args) is not dict:
        _args = vars(_args)
    if not _args['configuration_file']:
        raise FileNotFoundError
    with open(_args['configuration_file'], "r") as stream:
        config = yaml.safe_load(stream)

    main(config)

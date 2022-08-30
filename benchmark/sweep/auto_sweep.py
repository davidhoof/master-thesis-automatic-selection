import multiprocessing as mp
import os
import glob
import subprocess
import sys
import wandb
import logging
from argparse import ArgumentParser
import argparse
import yaml

def gpu_worker(args):
    gpu_id=args[0]
    q=args[1]
    
    # print(gpu_id)
    # print(q)
    
    while not q.empty():
        (sweep_id,number_of_runs) = q.get()  
        username,project,sweep_id=sweep_id.split("/")
        # print(username,project,sweep_id)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
        # !echo Hello
        wandb.agent(sweep_id,count=number_of_runs,project=project,entity=username)
        
        
def main(config):    
    
    root_folder=config['root']
    sweep_name=config['sweep_name']
    datasets=config['datasets']

    gpus = config['gpus']
    models = config['models']
    number_of_runs=config['number_of_runs']
    
    pool = mp.Pool(len(gpus))
    m = mp.Manager()
    
    queues={}
    for gpu, model in zip(gpus,models):
        queues[model]=m.Queue()    
    
    sweep_list=[]
    for dataset in datasets:
        with open(glob.glob(os.path.join(root_folder,f"{sweep_name}{dataset}","sweep_agent_commands*.txt"))[-1]) as file:
            lines = file.readlines()
            sweep_list = [(line.rstrip().split(" ")[-1], line.rstrip().split(" ")[-2]) for line in lines]
            
    for sweep in sweep_list:
        for model in models:
            if model == sweep[0]:
                 queues[model].put((sweep[1],number_of_runs))    
                    
    que_list=list(queues.values())

    processes = []
    
    processes=pool.map(gpu_worker,zip(gpus,que_list))
        
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

    
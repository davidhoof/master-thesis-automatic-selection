from argparse import ArgumentParser

import yaml

from result_preparation.save.sweep_extractor import SweepExtractor

if __name__ == "__main__":
    parser = ArgumentParser(description="Copy sweeps to an continuable result folder by the command list")
    parser.add_argument('configuration_file', help="Path of the configuration file")
    parser.add_argument('command_path', help="Path of the command file")

    _args = parser.parse_args()

    if type(_args) is not dict:
        _args = vars(_args)
    if not _args['configuration_file']:
        raise FileNotFoundError
    with open(_args['configuration_file'], "r") as stream:
        config = yaml.safe_load(stream)

    se = SweepExtractor(
        config['sweep_name'],
        config['root_path'],
        config['output_from'],
        config['output_to_path'],
        config['checkpoint_order'])

    se.copy_sweeps(_args['command_path'])

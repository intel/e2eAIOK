import init_hydro
from hydroai.hydroautolearner import HydroAutoLearner
import argparse
import sys

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='could be in-stock model name or udm(user-define-model)')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--conf', type=str, default='conf/hydroai_defaults.conf', help='hydroai defaults configuration')
    parser.add_argument('--executable_python', type=str, default='', help='user env python path')
    parser.add_argument('--program', type=str, default='', help='user defined train.py')
    parser.add_argument('--no_sigopt', dest="enable_sigopt", action="store_false", default=True, help='if disable sigopt')
    return parser.parse_args(args).__dict__

def main(input_args):
    learner = HydroAutoLearner(input_args)
    learner.submit()
    
    model = learner.get_best_model()
    print("\nWe found the best model! Here is the model explaination")
    model.explain()

if __name__ == '__main__':
    input_args = parse_args(sys.argv[1:])
    main(input_args)
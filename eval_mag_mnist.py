import os
import sys

import argparse
from glob import glob
import numpy as np

from greedy_alg.greedy_io import *
from common.io import _load, find_float_in_str

greedy_root = 'prob_regressor_results/greedy/mlp_mnist_10000samples_10000batches.new'
#greedy_root = 'mlp_mnist_10000samples_10000batches.new'

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate all greedy experiments')
    parser.add_argument('--gpu', default=0, type=int,
                        help='specify the GPU ID to run this experiments')
    return parser.parse_args()


def eval_greedy_all_exps(args, greedy_root):
    #sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_root)
    gpu = args.gpu 
    sparsity = np.append(np.arange(0.1, 0.9, 0.1), 0.85)
    for s in sparsity:
        os.system(f'sh scripts/mnist_tests.mag.sh {s} {gpu}')

if __name__ == '__main__':
    args = get_parser()
    eval_greedy_all_exps(args, greedy_root)


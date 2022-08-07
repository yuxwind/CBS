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
    parser.add_argument('--use-updated-weights', action='store_true', 
                        help='use the updated weights or not (default: False)')
    parser.add_argument('--gpus', default='0,', type=str,
                        help='specify the GPU ID to run this experiments')
    return parser.parse_args()


## eval all the pruned model by greedy algorithm
def eval_greedy_all_exps(greedy_root, args):
    #all_exps = get_greedy_all_exps(greedy_root, only_last_iter=True)
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_root)
    
    gpus = [int(i.strip()) for i in args.gpus.split(',') if i != '']
    gpu_idx = -1
    
	# debug
    sparsity = np.append(np.arange(0.1, 0.9, 0.1), 0.85)
    thresholds = [1e-4, 1e-05]
    ranges = [10, 100]
    max_no_matches = [10, 20]

    for s in sparsity:
        for m in max_no_matches:
            for r in ranges:
                for t in thresholds:
                    fpath = get_greedy_exp_paths(greedy_root, 
                                s, t, r,  m, 
                                only_last_iter=True)
                    if fpath == None or not os.path.exists(fpath):
                        continue           
                    gpu_idx += 1
                    gpu = gpus[gpu_idx % len(gpus)]
                    name = os.path.basename(fpath)
                    cfgs = find_float_in_str(name)
                    exp_name = get_greedy_exp_name(*cfgs[:-1])
                    itr  = 'iter_' + str(int(cfgs[-1]))
                    eval_path = exp_name + '.npy'
                    has_evaluated = False
                    if os.path.exists(eval_path):
                        eval_data = _load(eval_path).item() 
                        if itr in eval_data:
                            if args.use_updated_weights:
                                if 'top1.updated_weight' in eval_data[itr]:
                                    has_evaluated = True
                            else:
                                if 'top1' in eval_data[itr]:
                                    has_evaluated = True
                    if not has_evaluated:
                        if args.use_updated_weights:
                            ss = '--update-weights'
                        else:
                            ss = '--not-update-weights'
                        os.system(f'sh scripts/mnist_tests.comb.sh {fpath} {gpu} {ss}')

if __name__ == '__main__':
    args = get_parser()
    eval_greedy_all_exps(greedy_root, args)


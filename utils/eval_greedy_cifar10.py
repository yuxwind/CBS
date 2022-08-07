import os
import sys

import argparse
from glob import glob

from greedy_alg.greedy_io import *
from common.io import _load, find_float_in_str
from greedy_alg.cmpr_pruned import plot_acc_vs_swap_cnt

from env import ROOT,
from dir_cfg import greedy_root, eval_script
#greedy_root = 'prob_regressor_results/greedy/resnet20_cifar10_1000samples_1000batches_0seed/'
#greedy_root = 'prob_regressor_results/greedy/resnet20_cifar10_1000samples_1000batches_0seed_allweights'
#greedy_root = 'prob_regressor_results/greedy/resnet20_cifar10_1000samples_1000batches_0seed_allweights_init-mag'
#greedy_root = 'prob_regressor_results/greedy/resnet20_cifar10_1000samples_1000batches_0seed_init-mag'

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate all greedy experiments')
    parser.add_argument('--use-updated-weights', action='store_true', 
                        help='use the updated weights or not (default: False)')
    parser.add_argument('--gpus', default='0,', type=str,
                        help='specify the GPU ID to run this experiments')
    parser.add_argument('-t', default=1e-4, type=float)
    parser.add_argument('-s', default=0.6, type=float)
    
    return parser.parse_args()

## eval all the pruned model by magnitude 
def eval_mag_all_exps(greedy_root, args):
    #all_exps = get_greedy_all_exps(greedy_root, only_last_iter=True)
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_root)
    
    #sparisty = [0.2,0.4,0.6,0.8,0.9,0.98]
    #thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
    gpus = [int(i.strip()) for i in args.gpus.strip(',').split(',')]
    gpu_idx = -1

    eval_root = greedy_root + '.eval'
   
    for s in sparsity:
        for m in max_no_matches:
            for r in ranges:
                for t in thresholds:
                    name = get_greedy_exp_name(s, t, r, m)
                    fpath = os.path.join(greedy_root, name + '-magnitude.npy')
                    #import pdb;pdb.set_trace()
                    if not os.path.exists(fpath):
                            continue
                    gpu_idx += 1
                    gpu = gpus[gpu_idx % len(gpus)]
                    
                    eval_path = os.path.join(eval_root, name + '.npy')
                    key = 'magnitude'

                    has_evaluated = False
                    if os.path.exists(eval_path):
                        eval_data = _load(eval_path).item() 
                        if key in eval_data and 'top1' in eval_data[key]:
                            has_evaluated = True
                    if not has_evaluated:
                        ss = '--not-update-weights'
                        #os.system(f'bash scripts/sweep_cifar10_resnet20_oneshot_prune_all_weights.sh {fpath} {gpu} {ss}')
                        os.system(f'bash {eval_script} {fpath} {gpu} {ss}')
                    return

## eval all the pruned model by greedy algorithm
def eval_greedy_all_exps(greedy_root, args):
    #all_exps = get_greedy_all_exps(greedy_root, only_last_iter=True)
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_root)

    gpus = [int(i.strip()) for i in args.gpus.strip(',').split(',')]
    gpu_idx = -1
    max_iter = 1000

	# debug
    #sparsity = [0.4] 
    #thresholds = [1e-6]
    #ranges = [10]
    #max_no_matches = [10]
    
    sparsity = [args.s]
    thresholds = [1e-3, 1e-4, 1e-5]
    max_no_matches = [10,20]
    max_iter = 30
    
    only_last_iter = False
    
    for s in sparsity:
        for m in max_no_matches:
            for r in ranges:
                for t in thresholds:
                    fpaths = get_greedy_exp_paths(greedy_root, s, t, r,  m,
                                only_last_iter=only_last_iter)
                    if fpaths == None:
                        continue
                    if only_last_iter:
                        fpaths = [fpaths]
                    for fpath in fpaths:
                        if fpath == None or not os.path.exists(fpath):
                            continue
                        gpu_idx += 1
                        gpu = gpus[gpu_idx % len(gpus)]
                        exp_name = get_greedy_exp_name(s, t, r, m)
                        name = os.path.basename(fpath)
                        eval_root = os.path.dirname(fpath) + '.eval'
                        cfgs = find_float_in_str(name)
                        itr  = int(cfgs[-1])
                        
                        ## Just for job schdule
                        if itr > max_iter:
                            continue
                        ################
                        itr  = f'iter_{itr}'
                        eval_path = os.path.join(eval_root, exp_name + '.npy')
                        #import pdb;pdb.set_trace()
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
                        #import pdb;pdb.set_trace()
                        if not has_evaluated:
                            if args.use_updated_weights:
                                ss = '--update-weights'
                            else:
                                ss = '--not-update-weights'
                            os.system(f'bash {eval_script} {fpath} {gpu} {ss}')
                            #os.system(f'bash scripts/sweep_cifar10_resnet20_oneshot_prune_all_weights.sh {fpath} {gpu} {ss}')

                    #if not only_last_iter:
                    #    plot_acc_vs_swap_cnt(greedy_root, sparsity, T, arange, max_no_match)
if __name__ == '__main__':
    args = get_parser()
    eval_mag_all_exps(greedy_root, args)
    #eval_greedy_all_exps(greedy_root, args)

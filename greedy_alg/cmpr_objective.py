import os,sys
sys.path.append('..')

import glob

import torch
import numpy as np

from env import *
from common.io import _load, _load_attr, _dump
from greedy_options import get_parse
from greedy_io import extract_greedy_cfg
from greedyblock import GreedyBlockPruner

def load_pruned(fp, key=None):
    keys = ['pruned_idx', 'iter_time', 'accumulated_time', 'meta']
    meta_keys = ['obj_before_swap', 'obj_after_swap', 'reduction', 'swap_in_i', 'swap_out_j',
                 'swap_in_layer', 'swap_out_layer']
    if key is not None:
        assert(key in keys, f"{key} does not belong to [{keys}] in {fp}")
    return _load_attr(fp, key)

################################################################################
# cmpr the objective value of one result from greedy alg. 
#       with magnitude-based, woodfisherblock
###############################################################################
def cmpr_obj(sparsity, wgh_path, fp_greedy, args):
    #
    args.method = 'greedyblock'
    pruner = GreedyBlockPruner(sparsity, wgh_path=wgh_path, args=args)
    pruner.init_with_magnitude()
    obj_mag = pruner.cal_objective()
    print(f'obj_magnitude: {obj_mag}\n len(pruned_set): {len(pruner.set_pruned)}')
    pruner.init_with_wg()
    obj_wg  = pruner.cal_objective()
    print(f'obj_wg: {obj_wg}\n len(pruned_set): {len(pruner.set_pruned)}')
    pruner.init_with_woodfisherblock()
    obj_wf  = pruner.cal_objective()
    print(f'obj_woodfisherblock: {obj_wf}\n len(pruned_set): {len(pruner.set_pruned)}')
    
    pruner.set_pruned = load_pruned(fp_greedy, key='pruned_idx')
    obj_greedy = pruner.cal_objective()
    print(f'obj_greedyblock: {obj_greedy}\n len(pruned_set): {len(pruner.set_pruned)}')

################################################################################
# cmpr the objective value of all results from greedy alg. 
#       with magnitude-based, woodfisherblock
def cmpr_obj_greedy_dir(sparsity, wgh_path, fp_dir, args):
    args.method = 'greedyblock'
    pruner = GreedyBlockPruner(sparsity, wgh_path=wgh_path, args=args)
    pruner.init_with_magnitude()
    set_mag = pruner.set_pruned.clone().cpu().numpy()
    obj_mag = pruner.cal_objective()
    print(f'\n#######sparisyt: {sparsity}#########')
    print(f'obj_magnitude: {obj_mag}\n len(pruned_set): {len(pruner.set_pruned)}, ')
    pruner.init_with_wg()
    obj_wg  = pruner.cal_objective()
    print(f'obj_wg: {obj_wg}\n len(pruned_set): {len(pruner.set_pruned)}')
    pruner.init_with_woodfisherblock()
    obj_wf  = pruner.cal_objective()
    print(f'obj_woodfisherblock: {obj_wf}\n len(pruned_set): {len(pruner.set_pruned)},',
            f'{len(np.setdiff1d(set_mag, pruner.set_pruned.cpu().numpy()))}')

    for fp in sorted(glob.glob(os.path.join(fp_dir, f'sparsity{sparsity}*.npy')))[-2:-1]:
        pruner.set_pruned = load_pruned(fp, key='pruned_idx')
        obj_greedy = pruner.cal_objective()
        print('\t', os.path.basename(fp))
        print(f'obj_greedyblock: {obj_greedy} \n len(pruned_set): {len(pruner.set_pruned)},',
            f'{len(np.setdiff1d(set_mag, pruner.set_pruned.cpu().numpy()))}')
    
def ana_obj():
    args = get_parse()
    wgh_path  = args.wgh_path
    fp_dir = os.path.join(RESULT_ROOT,
                    "greedy_online-cifar10-resnet20_no_allweights",
                    "results.train_loss_all_samples",
                    "resnet20_cifar10_1000samples_1000batches_0seed")
    fp_greedy = os.path.join(fp_dir, "sparsity0.80_T1.00e-03_Range10_NOMATCH10-iter_1.npy")
    args.sparsity, args.threshold, args.range, args.max_no_match = extract_greedy_cfg(fp_greedy)
    fp_woodfisher = args.woodfisher_mask_path
    sparsity = args.sparsity
    wgh_path = args.wgh_path
    args.woodfisher_mask_path = os.path.join(WEIGHT_ROOT, 
                    "resnet20_cifar10_1000samples_1000batches_0seed.fisher_inv", 
                    f"global_mask_sparsity{sparsity}.pkl")
    cmpr_obj(sparsity, wgh_path, fp_greedy, args)

def ana_obj_greedy_dir():
    args = get_parse()
    wgh_path  = args.wgh_path
    fp_dir = os.path.join(RESULT_ROOT,
                    "greedy_online-cifar10-resnet20_no_allweights",
                    "results.train_loss_all_samples",
                    "resnet20_cifar10_1000samples_1000batches_0seed")
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    fp_woodfisher = args.woodfisher_mask_path
    wgh_path = args.wgh_path
    for sparsity in sparsities:
        args.woodfisher_mask_path = os.path.join(WEIGHT_ROOT, 
                    "resnet20_cifar10_1000samples_1000batches_0seed.fisher_inv", 
                    f"global_mask_sparsity{sparsity}.pkl")
        cmpr_obj_greedy_dir(sparsity, wgh_path, fp_dir, args)
if __name__ == '__main__':
    #ana_obj()
    ana_obj_greedy_dir()

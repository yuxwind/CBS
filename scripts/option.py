import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import shutil

def get_hyperparams(arch, greedy_option, scale_option):
    assert(arch in ['mlp', 'cifarnet', 'resnet20', 'resnet50'], f"arch {arch} not in ['mlp', 'cifarnet', 'resnet20', 'resnet50']!!!!!!!!!!")
    assert(greedy_option in ['all', 'one', 'placeholder', 'manual'], f"greedy_option {greedy_option} not in ['all', 'test', 'manual']!!!!!!!!!!!!!")
    assert(scale_option in ['all', 'one'], f"greedy_option {greedy_option} not in ['all', 'test', 'manual']!!!!!!!!!!!!!")
    # ResNet20
    sparsity_resnet20   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    # CifarNet
    sparsity_cifarnet   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    sparsity_cifarnet.extend([0.75, 0.857, 0.875, 0.933, 0.95, 0.967, 0.98, 0.986, 0.99]) # additional extreme sparsity for CIFARNet
    sparsity_resnet50   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sparsity_mobilenet   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sparsity_mlpnet   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    seeds = [0, 1, 2, 3, 4]

    if arch == 'cifarnet':
        sparsity = sparsity_cifarnet
    if arch == 'resnet20':
        sparsity = sparsity_resnet20
    if arch == 'mlpnet':
        sparsity = sparsity_mlpnet
    if arch == 'moblienet':
        sparsity = sparsity_mobilenet

    if greedy_option == 'all':
        #thresholds = [1e-3, 1e-4, 1e-5]
        #ranges     = [10, 100]
        #max_no_match=[10, 20]
        thresholds = [1e-3, 1e-4]
        ranges     = [10]
        max_no_match=[20]

    elif greedy_option == 'one': 
        sparsity   = [0.9]
        thresholds = [1e-5]
        ranges     = [10]
        max_no_match=[10]
    elif greedy_option == 'placeholder':
        sparsity   = [0.9]
        thresholds = [1e-5]
        ranges     = [10]
        max_no_match=[10]
    else:
        raise("!!!Please mually set the greedy_options in scripts/option.py")
    if scale_option == 'all':
        #scale_update=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scale_update=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else: 
        scale_update=[1.0]
    return sparsity, thresholds, ranges, max_no_match, scale_update, seeds

def run_options(args):
    parser = argparse.ArgumentParser(description='Run experiments')

    parser.add_argument('--script', type=str, help='script to run')
    parser.add_argument('--gpus', default='0', type=str,
                        help='gpus to run on')
    parser.add_argument('--paral_cnt', default=1, type=int,
                        help='paralle jobs on each GPU')
    parser.add_argument('--test', action='store_true', 
                        help='for test and debug')
    parser.add_argument('--update_scale_one', action=store_true, 
                        help='when update weights, run this firstly ')
    script = args.script
    archs = ['mlp', 'cifarnet', 'resnet20', 'resnet50']
    arch  = None
    for a in archs:
        if a in script:
            arch = a
   #TODO 
     

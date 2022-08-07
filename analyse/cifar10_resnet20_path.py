import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import shutil
import matplotlib.pyplot as plt

from common.time import now_
from env import *
from greedy_alg.greedy_io import get_greedy_exp_name
from common.io import _load,_dump
from analyse.load_results import *



def read_baseline():
    path = 'prob_regressor_results/cifar10_resnet20_baseline.txt'
    data = np.loadtxt(path)
    baseline_dict = dict(zip(data[:,0], data[:,1:]))
    return baseline_dict

def baseline():
    EXP_DIR=RESULT_ROOT
    EXP_NAME = 'sweep_cifar10_resnet20_woodfisherblock'
    WFB_DIR  = os.path.join(EXP_DIR, EXP_NAME)

    EXP_NAME = 'sweep_cifar10_resnet20_woodfisherblock_no_weight_update'
    WFB_no_update_DIR  = os.path.join(EXP_DIR, EXP_NAME)

    return WFB_DIR, WFB_no_update_DIR

def target_greedyonline_init_magperb():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_52'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"    

    
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    SINGLE_NOWDATE='results.update_w.single.20220113.02_16.scale_update_%s'
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.02_15.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

def target_greedyonline_init_mag():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_52'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"    

    
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    SINGLE_NOWDATE='results.update_w.single.20220113.02_16.scale_update_%s'
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.02_15.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

def target_greedyonline_init_wfb():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_59'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"

    EVAL_NOWDATE='train_loss_all_samples.20211202.22_59'
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    EVAL_NOWDATE='eval_update_w.20211207.14_58'
    EVAL_update_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def get_scales(methodToRun):
    # extract the scales from the paths of results
    GREEDY_DIR, EVAL_DIR, EVAL_update_DIR = methodToRun(0)
    pattern = os.path.dirname(EVAL_update_DIR).split('scale_update_')[0] + 'scale_update_*'
    scales  = sorted([float(d.split('scale_update_')[1]) for d in glob(pattern)])
    return scales

def target_greedyonline_init_wfb_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_59'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"

    EVAL_NOWDATE='train_loss_all_samples.20211202.22_59'
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    EVAL_NOWDATE=f'eval_update_w.20211209.00_55.scale_update_{scale_prune}'
    EVAL_update_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_wf_update_w_multiple_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-woodfisher-update_rm_multiple'
    NOWDATE=f'results.20211217.01_55_09.scale_update_{scale_prune}'
    
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_wf_update_w_single_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-woodfisher-update_rm_single'
    NOWDATE=f'results.20211217.01_54_18.scale_update_{scale_prune}'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_mag_update_w_multiple_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-mag-update_rm_multiple'
    NOWDATE=f'results.20211217.02_00_36.scale_update_{scale_prune}'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_random_update_w_multiple_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-random-update_rm_multiple'
    NOWDATE=f'results.20211221.23_33_30.scale_update_{scale_prune}'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_random_update_w_single_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-random-update_rm_single'
    NOWDATE=f'results.20211222.01_45_59.scale_update_{scale_prune}'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR
    
def get_scale_rst(methodToRun, sparsity, thresholds, ranges, max_no_match):
    scale_update = get_scales(methodToRun)
    N_scale = len(scale_update)
    scale_update_rst = []
    for i in range(N_scale):
        scale = scale_update[i]
        if scale == 0.0:
            scale = 0
        # collect reults for greedy with wfb as init_method
        #greedy_dir, eval_dir, eval_update_dir= target_greedyonline_init_wfb_scale_update(scale)
        greedy_dir, eval_dir, eval_update_dir= methodToRun(scale)
        undone = check_done(
                        sparsity, thresholds, ranges, max_no_match,
                        greedy_dir, eval_dir, eval_update_dir)
        if undone:
            return
        all_select_wfb_s, all_update_wfb_s, unpruned_acc_s, best_select_wfb_s, best_update_wfb_s = get_eval(
                        sparsity, thresholds, ranges, max_no_match,
                        greedy_dir, eval_dir, eval_update_dir)
        scale_update_rst.append([best_update_wfb_s[s] for s in sparsity])
        #if i == 0:
        #    scale_update_rst.append([best_select_wfb[s] for s in sparsity])
    #scale_update_rst.append([best_update_wfb[s] for s in sparsity])
    scale_update_rst = np.array(scale_update_rst)
    return scale_update, scale_update_rst

def run():
    pass
if __name__ == '__main__':
    #check_done()
    run()

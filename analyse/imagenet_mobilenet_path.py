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
    path = 'prob_regressor_results/imagenet_mobilenet_baseline.txt'
    data = np.loadtxt(path)
    baseline_dict = dict(zip(data[:,0], data[:,1:]))
    return baseline_dict

def baseline():
    EXP_DIR=RESULT_ROOT
    EXP_NAME = 'sweep_imagenet_mobilenet_woodfisherblock'
    WFB_DIR  = os.path.join(EXP_DIR, EXP_NAME)

    EXP_NAME = 'sweep_imagenet_mobilenet_woodfisherblock_no_weight_update'
    WFB_no_update_DIR  = os.path.join(EXP_DIR, EXP_NAME)

    return WFB_DIR, WFB_no_update_DIR

#TODO
def target_greedyonline_init_magperb():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_52'
    ARCH_NAME="mobilenet_imagenet_960000samples_400batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"    

    
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    SINGLE_NOWDATE='results.update_w.single.20220113.02_16.scale_update_%s'
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'

    #TO modify
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.02_15.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

#TODO
def target_greedyonline_init_mag():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_52'
    ARCH_NAME="mobilenet_imagenet_960000samples_400batches_%dseed"
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
    EXP_NAME='imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_59'
    ARCH_NAME="mobilenet_imagenet_960000samples_400batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"

    EVAL_NOWDATE='train_loss_all_samples.20211202.22_59'
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    EVAL_NOWDATE='eval_update_w.20211207.14_58'
    EVAL_update_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

if __name__ == '__main__':
    #check_done()
    run()

import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import shutil
import matplotlib.pyplot as plt

from common.time import now_
from env import *
from greedy_alg.greedy_io import get_greedy_exp_name
from common.io import _load,_dump, parent_dir
from analyse.load_results import *
from scripts.option import get_hyperparams
from analyse.ana_greedyonline import *


#TODO: rewrite for cifarnet
def baseline():
    EXP_DIR=RESULT_ROOT
    EXP_NAME = 'sweep_cifar10_cifarnet_woodfisherblock'
    #NOWDATE='log.20220124.20_32_39'
    #ARCH_NAME= f"cifarnet_cifar10_%dseed"
    #WFB_DIR  = os.path.join(EXP_DIR, EXP_NAME, NOWDATE, ARCH_NAME)
    WFB_DIR  = os.path.join(EXP_DIR, EXP_NAME)
    #WFB_PATH = os.path.join() 

    EXP_NAME = 'sweep_cifar10_cifarnet_woodfisherblock_no_weight_update'
    #NOWDATE='log.20220125.05_02_36'
    #ARCH_NAME= f"cifarnet_cifar10_%dseed"
    WFB_no_update_DIR  = os.path.join(EXP_DIR, EXP_NAME)
    
    #mag_rst  = load_txt(os.path.join(MAG_DIR, 'eval_result.txt'))
    #wfb_rst  = load_txt(os.path.join(WFB_DIR, 'eval_result.txt'))
    #wfb_no_update_rst  = load_txt(os.path.join(WFB_no_update_DIR, 'eval_result.txt'))
    #
    #sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    
    #return dict(zip(sparsity, mag_rst), dict(zip(sparsity,wfb_rst), dict(zip(sparsity, wfb_no_update_rst)
    #return mag_rst, wfb_rst, wfb_no_update_rst
    return WFB_DIR, WFB_no_update_DIR

def target_cifarnet_v1():
    EXP_DIR=RESULT_ROOT
    dataset= 'cifar10'
    arch   = 'cifarnet'
    fisher_subsample_batches= 1000
    sample_in_one_batch = 1
    seed  = 0
    # for init=mag
    EXP_NAME  = f'{dataset}-{arch}-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    NOWDATE   = 'results.train_loss_all_samples.20220112.03_57_26'
    # for init=woodfish
    EXP_NAME  = f'{dataset}-{arch}-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE   = 'results.train_loss_all_samples.20220112.03_55_42'
    
    N_samples = fisher_subsample_batches * sample_in_one_batch
    ARCH_NAME = f"{arch}_{dataset}_{N_samples}samples_{fisher_subsample_batches}batches_{seed}seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"
    return GREEDY_DIR

def target_greedyonline_init_mag():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    NOWDATE='results.train_loss_all_samples.20220112.03_57_26'
    ARCH_NAME="cifarnet_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"    

    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}.eval'

    SINGLE_NOWDATE='results.update_w.single.20220113.02_16.scale_update_%s' 
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'
    
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.02_15.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

def target_greedyonline_init_magperb():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers'
    NOWDATE='results.train_loss_all_samples.20220112.03_57_26'
    ARCH_NAME="cifarnet_cifar10_1000samples_1000batches_%dseed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"    

    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}.eval'

    SINGLE_NOWDATE='results.update_w.single.20220113.02_16.scale_update_%s' 
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'
    
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.02_15.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

def taget_greedyonline_longclaw():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    NOWDATE='results.train_loss_all_samples.20220111.00_38_57'
    ARCH_NAME="cifarnet_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR_mag=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"    

    EVAL_DIR_mag=f'{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}.eval'
     
    EXP_NAME='cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='reults.train_loss_all_samples.20220112.04_00_27'
    GREEDY_DIR_woodfisher=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"    
    
    EVAL_DIR_woodfisher=f'{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}.eval'
    return GREEDY_DIR_mag, EVAL_DIR_mag, GREEDY_DIR_woodfisher, EVAL_DIR_woodfisher 

def target_greedyonline_init_wfb():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='results.train_loss_all_samples.20220112.03_55_42'
    ARCH_NAME="cifarnet_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}.eval'
    
    SINGLE_NOWDATE='results.update_w.single.20220113.21_31_00.scale_update_%s' 
    DIR_single_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{SINGLE_NOWDATE}/{ARCH_NAME}.eval'
    
    MULTIPLE_NOWDATE='results.update_w.multiple.20220113.21_12_30.scale_update_%s'
    DIR_multiple_update_fmt=f'{EXP_DIR}/{EXP_NAME}/{MULTIPLE_NOWDATE}/{ARCH_NAME}.eval'
    
    return GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt

def target_ablation_study():
    EXP_DIR=RESULT_ROOT
    EXP_NAME= ''
    NOWDATE=  ''
    ARCH_NAME="cifarnet_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/{NOWDATE}/{ARCH_NAME}"

def read_baseline():
    path = 'prob_regressor_results/cifar_baseline.txt'
    data = np.loadtxt(path)
    baseline_dict = dict(zip(data[:,0], data[:,1:])) 
    return baseline_dict


def collect_results():
    GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt = target_greedyonline_init_mag() 
    sparsity, thresholds, ranges, max_no_match, scale_update = get_hyperparams('cifarnet', 'all', 'all') 
    greedy_dict, greedy_arr = get_eval_greedy(sparsity, thresholds, ranges, max_no_match, EVAL_DIR)
    single_dict, single_arr = get_eval_update(sparsity, thresholds, ranges, max_no_match, EVAL_DIR, 
                                DIR_single_update_fmt%1.0)
    multi_dict, multi_arr = get_eval_update(sparsity, thresholds, ranges, max_no_match, EVAL_DIR, 
                                DIR_multiple_update_fmt%1.0)
    baseline_dict = read_baseline()
    arr = np.concatenate([greedy_arr, single_arr[:,5:], multi_arr[:,5:]], axis=1)
    N1,N2=arr.shape
    arr1 = [] 
    for i in range(N1):
        dd = baseline_dict[(float(arr[i,1]))][1:]
        aa = []
        for j in range(6):
            aa.append(str(dd[j]))
        arr1.append(aa)    
    import pdb;pdb.set_trace()
    arr = np.concatenate([arr, np.array(arr1)], axis=1)

    save_path = parent_dir(GREEDY_DIR, times=2) + '.txt'
    import pdb;pdb.set_trace()
    np.savetxt(save_path, arr, fmt='%s',
            header = 'greedy_path sparsity threshold range max_no_match ' +
                     'greedy_top1, greedy_top5 best_iter ' +
                     'single_update_top1 single_update_top5 ' +
                     'multiple_update_top1 multiple_update_top5'
                     'woodfisher_top1, woodfisehr_top5 ' +
                     'woodfisher_select_top1, woodfisher_select_top5 ' +
                     'mag_top1, mag_top5 '
    ) 


def cmp_greedy_on_bluefish_longclaw():
    GREEDY_DIR_mag, EVAL_DIR_mag, GREEDY_DIR_woodfisher, EVAL_DIR_woodfisher = taget_greedyonline_longclaw()
    sparsity, thresholds, ranges, max_no_match, scale_update = get_hyperparams('cifarnet', 'all', 'all') 
    greedy_dict, greedy_arr = get_eval_greedy(sparsity, thresholds, ranges, max_no_match, EVAL_DIR_mag)
    
    import pdb;pdb.set_trace() 
    #save_path = parent_dir(GREEDY_DIR_mag, times=2) + '.txt'
    #bf_rst = np.loadtxt(save_path, fmt='%s')
    #bf_strm= bf_rst[:,1:5]
    #lc_rst = greedy_arr
    #lc_strm= lc_rst[:,1:5]
    #assert( (lc_strm == bf_strm).all(), "Not matching!!!")
    #arr = np.concatenate([bf_rst[:,1:5], lc_rst[5:7]], axis=1)
    #save_path = parent_dir(GREEDY_DIR_mag, times=2) + '-cmpr.txt'
    arr = greedy_arr[:, 1:]
    save_path = parent_dir(GREEDY_DIR_mag, times=2) + '-1.txt'
    np.savetxt(save_path, arr, fmt='%s',
            header = 'sparsity threshold, range max_no_match ' +
                     'greedy_top1.lc greedy_top5.lc' 
    ) 


def run():
    pass
if __name__ == '__main__':
    #check_done()
    #collect_results()
    #cmp_greedy_on_bluefish_longclaw()
    WFB_DIR, WFB_no_update_DIR = baseline()
    from analyse.ana_greedyonline import collect_baseline
    rst1 = collect_baseline(WFB_DIR)

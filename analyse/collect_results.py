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

arch = 'resnet20'

if arch == 'mlpnet':
    from analyse.mnist_mlpnet_path import baseline, read_baseline
    from analyse.mnist_mlpnet_path import target_greedyonline_init_mag
    from analyse.mnist_mlpnet_path import target_greedyonline_init_magperb
    save_path = 'prob_regressor_results/mnist-mlpnet.txt'
if arch == 'cifarnet':
    from analyse.cifar10_cifarnet_path import baseline, read_baseline
    from analyse.cifar10_cifarnet_path import target_greedyonline_init_mag
    from analyse.cifar10_cifarnet_path import target_greedyonline_init_magperb
    save_path = 'prob_regressor_results/cifar10-cifarnet.txt'
if arch == 'resnet20':
    from analyse.cifar10_resnet20_path import baseline, read_baseline
    from analyse.cifar10_resnet20_path import target_greedyonline_init_mag
    from analyse.cifar10_resnet20_path import target_greedyonline_init_magperb
    save_path = 'prob_regressor_results/cifar10-resnet20.txt'

def collect_seed_results(seed):
    # greedy: init=mag
    GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt = target_greedyonline_init_mag() 
    sparsity, thresholds, ranges, max_no_match, scale_update, seeds = get_hyperparams('mlpnet', 'all', 'all') 
   
    # arr = [eval_path, s, t, r, m, top1, top5, best_iter, best_iter_sample_loss]
    # rst_dict[s][t][r][m] = [top1, top5, best_iter, best_iter_sample_loss]
    greedy_dict, greedy_arr = get_eval_greedy(sparsity, thresholds, ranges, max_no_match, EVAL_DIR%seed)
    
    multi_dict, multi_arr = get_eval_update(sparsity, thresholds, ranges, max_no_match, EVAL_DIR%seed, 
                                DIR_multiple_update_fmt%(1.0, seed))
    
    # greedy: init=mag_perb
    GREEDY_DIR, EVAL_DIR, DIR_single_update_fmt, DIR_multiple_update_fmt = target_greedyonline_init_magperb() 
    sparsity, thresholds, ranges, max_no_match, scale_update, seeds = get_hyperparams('mlpnet', 'all', 'all') 
    
    greedy_dict_perb, greedy_arr_perb = get_eval_greedy(
                sparsity, thresholds, ranges, max_no_match, EVAL_DIR%seed)
    
    multi_dict_perb, multi_arr_perb = get_eval_update(
                sparsity, thresholds, ranges, max_no_match, EVAL_DIR%seed, 
                DIR_multiple_update_fmt%(1.0, seed))
    
    # mag results # rst[sparsity]
    mag_dict = read_baseline()
    
    # woodfisher results
    WFB_DIR, WFB_no_update_DIR = baseline() 
    wfb_rst =collect_baseline(WFB_DIR) # rst[seed][sparsity]
    wfb_no_update_rst =collect_baseline(WFB_no_update_DIR)
     
    our_arr = np.concatenate([greedy_arr, multi_arr[:,5:], greedy_arr_perb[:,5:], multi_arr_perb[:,5:]], axis=1)
   
    # Repeat the baseline values, will append them to our_arr
    N = our_arr.shape[0]
    arr_baseline = [] 
    for i in range(N):
        sp = float(our_arr[i,1])
        bs = [seed] # seed, mag_top1, mag_top5, wfb_top1, wfb_top5, wfb_s_top1, wfb_s_top5, 
        try:
            bs.extend(mag_dict[sp])
        except:
            bs.extend(['', ''])
        try:
            bs.extend([
                wfb_rst[seed][sp]['top1'], 
                wfb_rst[seed][sp]['top5']])
        except:
            bs.extend(['',''])
        try:
            bs.extend([
                wfb_no_update_rst[seed][sp]['top1'], 
                wfb_no_update_rst[seed][sp]['top5']])
        except:
            bs.extend(['', ''])
        arr_baseline.append(bs)    
    arr_baseline = np.array(arr_baseline)

    arr = np.concatenate([our_arr,arr_baseline], axis=1)
    return arr


def collect_results():
    sparsity, thresholds, ranges, max_no_match, scale_update, seeds = get_hyperparams(arch, 'all', 'all') 
    all_rsts = []
    for seed in seeds:
        all_rsts.append(collect_seed_results(seed))
    all_rsts = np.concatenate(all_rsts)

    
    import pdb;pdb.set_trace()
    np.savetxt(save_path, all_rsts, fmt='%s',
            header = 'greedy_path sparsity threshold range max_no_match ' +
                     'LS_top1, LS_top5 LS_best_iter LS_sample_loss ' +
                     'LS_multiple_top1 LS_multiple_top5 ' +
                     'RMP_LS_top1 RMP_LS_top5 RMP_LS_best_iter RMP_LS_sample_loss ' +
                     'RMP_LS_multiple_top1 RMP_LS_multiple_top5 ' +
                     'seed ' +
                     'mag_top1 mag_top5 ' +
                     'WF_top1 WF_top5 ' +
                     'WF_select_top1, WF_select_top5 '
    )   

if __name__ == '__main__':
    #check_done()
    collect_results()
    #cmp_greedy_on_bluefish_longclaw()
    
    #WFB_DIR, WFB_no_update_DIR = baseline()
    #from analyse.ana_greedyonline import collect_baseline
    #rst1 = collect_baseline(WFB_DIR)

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
from analyse.ana_greedyonline import *

def target_greedyonline_init_wfb_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_59'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"

    EVAL_NOWDATE='train_loss_all_samples.20211202.22_59'
    EVAL_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    EVAL_NOWDATE=f'eval_update_w.20211209.00_55.scale_update_{scale_prune}'
    EVAL_update_DIR=f'{EXP_DIR}/{EXP_NAME}/results.{EVAL_NOWDATE}/{ARCH_NAME}.eval'

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def target_greedyonline_update_w_scale_update(scale_prune=0.1):
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-woodfisher-update_rm_multiple'
    NOWDATE=f'20211217.01_55_09.scale_update_{scale_prune}'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"

    EVAL_DIR=f'{GREEDY_DIR}.eval'
    EVAL_update_DIR=EVAL_DIR

    return GREEDY_DIR, EVAL_DIR, EVAL_update_DIR

def run():
    now = now_()
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    #thresholds = [1e-3, 1e-4, 1e-5]
    #ranges     = [10, 100]
    #max_no_match=[10, 20]
    #sparsity   = [0.8]
    thresholds = [1e-5]
    ranges     = [10]
    max_no_match=[10]
    scale_update=[-0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_scale     = len(scale_update)
    
    # collect baseline results
    mag_rst, wfb_rst, wfb_no_update_rst = baseline() 
    mag_rst = mag_rst[:,0]
    wfb_rst = wfb_rst[:,0]
    wfb_no_update_rst = wfb_no_update_rst[:,0]

    # collect reults for greedy with mag as init_method
    greedy_dir, eval_dir, eval_update_dir= target_greedyonline_init_mag()
    undone = check_done(
                    sparsity, thresholds, ranges, max_no_match,
                    greedy_dir, eval_dir, eval_update_dir)
    if undone:
        return
    all_select_mag, all_update_mag, unpruned_acc, best_select_mag, best_update_mag = get_eval(
                    sparsity, thresholds, ranges, max_no_match,
                    greedy_dir, eval_dir, eval_update_dir)

    # collect reults for greedy with woodfisherblock as init_method
    greedy_dir, eval_dir, eval_update_dir= target_greedyonline_init_wfb()
    undone = check_done(
                    sparsity, thresholds, ranges, max_no_match,
                    greedy_dir, eval_dir, eval_update_dir)
    if undone:
        return
    all_select_wfb, all_update_wfb, unpruned_acc, best_select_wfb, best_update_wfb = get_eval(
                    sparsity, thresholds, ranges, max_no_match,
                    greedy_dir, eval_dir, eval_update_dir)
    
    # set plots 
    x1 = np.array([0.03, 0.02, 0.01])
    x2 = np.array([0.01, 0.02, 0.03])
    eval_colors = {'r10_m10': 'r', 'r10_m20': 'g', 'r100_m10': 'b', 'r100_m20': 'c'}
    greedy_colors = {'r10_m10': 'r', 'r10_m20': 'g', 'r100_m10': 'b', 'r100_m20': 'c'}
   
    fig, ax = plt.subplots(1+len(sparsity))
    fig.set_size_inches(6,40)

    ax1 = ax[0]
    color='tab:red'
    ax1.set_xlabel('Target sparsity level')
    ax1.set_ylabel('test_acc (%)', color=color)
    
    # plot for baselines
    ax1.plot(sparsity, [unpruned_acc[s] for s in sparsity], 'k*-', label='uncompressed model')
    ax1.plot(sparsity, mag_rst, 'bo-.', label='magnitude')
    ax1.plot(sparsity, wfb_no_update_rst, 'go-.', label='woodfisherblock_no_update_w')
    ax1.plot(sparsity, wfb_rst, 'ro-', label='woodfisherblock')
   
    # plot for init_method=mag
    ax1.plot(sparsity, [best_select_mag[s] for s in sparsity], 'c^-.', 
                label = 'greedy-init_mag-no_update_w')
    ax1.plot(sparsity, [best_update_mag[s] for s in sparsity], 'm^-', 
                label = 'greedy-init_mag-update_w')
    # plot for init_method=woodfisherblock
    ax1.plot(sparsity, [best_select_wfb[s] for s in sparsity], '^-.', color='tab:blue',
                label = 'greedy-init_woodfisherblock-no_update_w')
    ax1.plot(sparsity, [best_update_wfb[s] for s in sparsity], '^-', color='tab:pink',
                label = 'greedy-init_woodfisherblock-update_w')

    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.04))
    #ax1.legend(loc='best')
        
    scale_update_rst = []
    for i in range(N_scale):
        scale = scale_update[i]
        # collect reults for greedy with mag as init_method
        #greedy_dir, eval_dir, eval_update_dir= target_greedyonline_init_wfb_scale_update(scale)
        greedy_dir, eval_dir, eval_update_dir= target_greedyonline_update_w_scale_update(scale)
        undone = check_done(
                        sparsity, thresholds, ranges, max_no_match,
                        greedy_dir, eval_dir, eval_update_dir)
        if undone:
            return
        all_select_wfb_s, all_update_wfb_s, unpruned_acc_s, best_select_wfb_s, best_update_wfb_s = get_eval(
                        sparsity, thresholds, ranges, max_no_match,
                        greedy_dir, eval_dir, eval_update_dir)
        scale_update_rst.append([best_update_wfb_s[s] for s in sparsity])
        if i == 0:
            scale_update_rst.append([best_select_wfb[s] for s in sparsity])
    scale_update_rst.append([best_update_wfb[s] for s in sparsity])
    scale_update_rst = np.array(scale_update_rst)
     
    scale_update.insert(1, 0.0)
    scale_update.append(1.0)
    for i, s in enumerate(sparsity):
        ax1 = ax[i+1]
        color='tab:red'
        ax1.set_xlabel('scaling factor multiplied to the weight update')
        ax1.set_ylabel('test_acc (%)', color=color)
        
        # plot for baselines
        #ax1.axhline(unpruned_acc[s],color='k', linestyle='-', marker='*',   
        #        xmax=max(scale_update), 
        #        label='uncompressed model')
        ax1.axhline(mag_rst[i], 
                color='b', linestyle='-.', #marker='o',
                xmin=min(scale_update), xmax=max(scale_update), 
                label=f'magnitude with sparsity={s}')
        ax1.axhline(wfb_rst[i],  
                color='r', linestyle='-', #marker='o',
                xmin=min(scale_update), xmax=max(scale_update), 
                label=f'woodfisherblock with sparsity={s}')
   
        # plot for init_method=woodfisherblock
        ax1.plot(scale_update, scale_update_rst[:,i], '^-',
                    label = f'greedy-init_woodfisherblock-update_w, sparisty={s}')
        ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.04))
        #ax1.legend(loc="best")
        
    fig.tight_layout()
    fig_p = os.path.join(PLOT_ROOT,  greedy_dir.split('/')[-3] + f'.scale_update.pdf')
    if os.path.exists(fig_p):
        os.rename(fig_p, fig_p.strip('.pdf') + f'.{now}.pdf')
    plt.savefig(fig_p)
    plt.close()

if __name__ == '__main__':
    #check_done()
    run()

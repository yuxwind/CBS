import os
import sys
sys.path.append('..')

import numpy as np
import torch 
from os.path import join as pjoin
import matplotlib.pyplot as plt

from common.io import _load
from greedy_alg.greedy_io import *

from env import ROOT, WEIGHT_ROOT
from env_cfg import greedy_root

def cmpr_mag(greedy_root, sparsity, T, arange, max_no_match, include_mag=True):
    # load weights and rank it
    exp_name = os.path.basename(greedy_root)
    
    name  = get_greedy_exp_name(sparsity, T, arange,  max_no_match) 
    ana_root = greedy_root + '.analysis'
    if not os.path.exists(ana_root):
        os.makedirs(ana_root)
    ana_name = pjoin(ana_root , name + '.npy')
    #if os.path.exists(ana_name):
    #    return None,None

    weight_data = _load(pjoin(WEIGHT_ROOT, exp_name + '.pkl'))
    weights = weight_data['W']
    sorted_idx = np.argsort(-np.absolute(weights))
    idx_2_rank = dict(zip(sorted_idx, range(len(weights))))
    
    #pruned_idxs = get_pruned_idx_for_all_iters(greedy_root, sparsity, T, arange,  max_no_match,
    #                include_mag)
    try:
        pruned_idxs, obj = get_greedy_for_all_iters(greedy_root, sparsity, T, arange,  max_no_match,
                        include_mag)
        top1, top1_train, loss_train = get_pruned_acc_for_all_iters(greedy_root, sparsity, T, arange,  
                        max_no_match, include_mag)
    except:
        return None, None, None
    evals = [top1, top1_train, loss_train]
    if evals is None:
        return None, None, None

   # load the pruned_idx of the magnitude-based pruning
    mag_pruned  = pruned_idxs[0] 
    mag_remained = np.setdiff1d(range(len(weights)), mag_pruned)
    
    remained_sets = [mag_remained]
    diff_sets = []
    diff_ranks = []
    N_diff_iters = []

    #print(f'N_mag_pruned\tN_comb_pruned\tN_diff\ttop1')
    for i in range(len(top1)):
        comb_pruned = pruned_idxs[i] 
        comb_remained = np.setdiff1d(range(len(weights)), comb_pruned)
    
        diff = np.setdiff1d(comb_remained, mag_remained)
        diff_rank = [idx_2_rank[i] for i in diff]
        #print(f'{len(mag_pruned)}\t{len(comb_pruned)}\t{len(diff)}\t{top1[i]}')
        remained_sets.append(comb_remained)
        diff_sets.append(diff)
        diff_ranks.append(diff_rank)
        N_diff_iters.append(len(diff))
        #import pdb;pdb.set_trace()
    np.save(ana_name, {'remained_sets': remained_sets, 'diff_sets': diff_sets, 'diff_ranks': diff_ranks})
    return evals, obj,  N_diff_iters

def plot_greedy(greedy_root, name, evals, obj, diff_cnt, include_mag=True):
    #test_acc = np.loadtxt('test_acc.txt')
    #diff_cnt = np.loadtxt('cmpr_mag.txt')
    if len(obj) == 0:
        return
    test_acc, train_acc, train_loss = evals
    iters = np.arange(len(test_acc))

    obj = obj[:len(test_acc)]
    diff_cnt = diff_cnt[:len(test_acc)]

    #import pdb;pdb.set_trace()
    if not include_mag: 
        # include_mag: start from 0; otherwise, start from 1
        iters += 1 

    fig, ax = plt.subplots(2)
    
    ax1 = ax[0]
    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('acc (%)', color=color)
    ax1.plot(iters,test_acc, color=color, label='test_acc')
   # ax1.plot(iters,train_acc, color='green', label='train_acc')
    ax1.tick_params(axis='y', labelcolor=color)
    
    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    #color = 'tab:blue'
    #ax2.set_ylabel('comb. - mag.', color=color)  # we already handled the x-label with ax1
    #ax2.plot(iters, diff_cnt, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('train_acc', color=color)  # we already handled the x-label with ax1
    ax2.plot(iters, train_acc, color=color, label='train_acc')
    ax2.tick_params(axis='y', labelcolor=color)
   
    ax3 = ax[1]
    color= 'tab:red'
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('obj_values', color=color)
    ax3.plot(iters, obj, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    ax4 = ax3.twinx()

    color = 'tab:blue'
    ax4.set_ylabel('train_loss', color=color)  # we already handled the x-label with ax1
    ax4.plot(iters, train_loss, color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig_root = os.path.join(greedy_root + '.analysis')
    if not os.path.exists(fig_root):
        os.makedirs(fig_root)
    plt.savefig(os.path.join(fig_root, f'{name}.pdf'))
    plt.close()

def plot_acc_vs_swap_cnt(greedy_root, sparsity, T, arange, max_no_match, include_mag=True):
    evals, obj, diff_cnt = cmpr_mag(greedy_root, sparsity, T, arange, max_no_match, include_mag) 
    name  = get_greedy_exp_name(sparsity, T, arange,  max_no_match) 
    if evals is None:
        print(f'evaluations are not done: {name}')
    else:
        #import pdb;pdb.set_trace()
        plot_greedy(greedy_root, name, evals, obj, diff_cnt)

def plot_acc_vs_swap_cnt_for_epxs(greedy_root, include_mag):
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_root)
    for s in sparsity:
        for m in max_no_matches:
            for r in ranges:
                for t in thresholds:
                    plot_acc_vs_swap_cnt(greedy_root, s, t, r, m, include_mag)

if __name__ == '__main__':
    #exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed'
    #exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed_allweights'
    #exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed_allweights_init-mag-test_fc'
    exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed-swap_one_False-init-mag'
    greedy_root = os.path.join(ROOT,'prob_regressor_results', 'greedyblock', exp_name)
    #sparsity = 0.8
    #T        = 1e-3
    #arange   = 10
    #max_no_match = 10
    #plot_acc_vs_swap_cnt(greedy_root, sparsity, T, arange, max_no_match)

    #exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed_init-mag'
    #exp_name = 'resnet20_cifar10_1000samples_1000batches_0seed_allweights_init-mag'
    #greedy_root = os.path.join(greedy_dir, exp_name)
    plot_acc_vs_swap_cnt_for_epxs(greedy_root, include_mag=True)

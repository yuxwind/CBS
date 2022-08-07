import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import shutil
import matplotlib.pyplot as plt
from glob import glob

from common.time import now_
from env import *
from greedy_alg.greedy_io import get_greedy_exp_name
from common.io import _load,_dump
from analyse.load_results import *
from analyse.ana_greedyonline import *
from analyse.ana_scale_update import target_greedyonline_update_w_scale_update
from common.utils  import SortedSuperSet

from analyse.cifar10_resnet_path import *

def get_scales(methodToRun):
    # extract the scales from the paths of results
    GREEDY_DIR, EVAL_DIR, EVAL_update_DIR = methodToRun(0)
    pattern = os.path.dirname(EVAL_update_DIR).split('scale_update_')[0] + 'scale_update_*'
    scales  = sorted([float(d.split('scale_update_')[1]) for d in glob(pattern)])
    return scales

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
        
    scales_greedy, scale_update_rst                  = get_scale_rst(
            target_greedyonline_update_w_scale_update,
            sparsity, thresholds, ranges, max_no_match,)
    scales_wf_multiple, scale_update_rst_wf_multiple= get_scale_rst(
            target_wf_update_w_multiple_scale_update,
            sparsity, thresholds, ranges, max_no_match,)
    scales_wf_single, scale_update_rst_wf_single    = get_scale_rst( 
            target_wf_update_w_single_scale_update,
            sparsity, thresholds, ranges, max_no_match,) 
    scales_mag_multiple, scale_update_rst_mag_multiple = get_scale_rst( 
            target_mag_update_w_multiple_scale_update,
            sparsity, thresholds, ranges, max_no_match,)
    scales_random_multiple, scale_update_rst_random_multiple = get_scale_rst( 
            target_random_update_w_multiple_scale_update,
            sparsity, thresholds, ranges, max_no_match,)
    scales_random_single, scale_update_rst_random_single = get_scale_rst( 
            target_random_update_w_single_scale_update,
            sparsity, thresholds, ranges, max_no_match,)
    scales_superset = SortedSuperSet()
    scales_superset.update(scales_greedy)
    scales_superset.update(scales_wf_multiple)
    scales_superset.update(scales_wf_single)
    scales_superset.update(scales_mag_multiple)
    scales_superset.update(scales_random_multiple)
    scales_superset.update(scales_random_single)

    #scale_update.insert(1, 0.0)
    #scale_update.append(1.0)
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
                xmin=scales_superset.min, xmax=scales_superset.max, 
                label=f'magnitude, sparsity={s}')
        ax1.axhline(wfb_rst[i],  
                color='g', linestyle='-', #marker='o',
                xmin=scales_superset.min, xmax=scales_superset.max, 
                label=f'woodfisherblock, sparsity={s}')
        ax1.axhline(best_select_mag[s],  
                color='r', linestyle='-', #marker='o',
                xmin=scales_superset.min, xmax=scales_superset.max, 
                label=f'greedy selection init=magnitude, sparsity={s}')
        ax1.axhline(best_select_wfb[s],  
                color='c', linestyle='-', #marker='o',
                xmin=scales_superset.min, xmax=scales_superset.max, 
                label=f'greedy selection init=woodfisher, sparsity={s}')

        all_values = SortedSuperSet()
        all_values.update([mag_rst[i], wfb_rst[i], best_select_mag[s], best_select_wfb[s]])
        
        # plot for init_method=woodfisherblock
        ax1.plot(scales_greedy, scale_update_rst[:,i], '^-',
                    label = f'greedy-update_rm_multiple, sparisty={s}')
        ax1.plot(scales_wf_multiple, scale_update_rst_wf_multiple[:,i], 'o-',
                    label = f'woodfisher-update_rm_multiple, sparisty={s}')
        ax1.plot(scales_wf_single, scale_update_rst_wf_single[:,i], '*-',
                    label = f'woodfisehr-update_rm_single, sparisty={s}')
        ax1.plot(scales_mag_multiple, scale_update_rst_mag_multiple[:,i], 'o-',
                    label = f'mag-update_rm_multiple, sparisty={s}')
        ax1.plot(scales_random_multiple, scale_update_rst_random_multiple[:,i], 'o-',
                    label = f'random-update_rm_multiple, sparisty={s}')
        ax1.plot(scales_random_single, scale_update_rst_random_single[:,i], '*-',
                    label = f'random-update_rm_single, sparisty={s}')
        all_values.update(scale_update_rst[:,i])
        all_values.update(scale_update_rst_wf_multiple[:,i])
        all_values.update(scale_update_rst_wf_single[:,i])
        all_values.update(scale_update_rst_mag_multiple[:,i])
        all_values.update(scale_update_rst_random_multiple[:,i])
        all_values.update(scale_update_rst_random_single[:,i])
        min_,max_  = all_values.min, all_values.max

        ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.04))
        #ax1.legend(loc="best")
        interval = max(0.5, int((max_ - min_)/5 * 2.0) / 2.0) 
        ax1.set_yticks(np.arange(int(min_ * 2 - 1.0)/2.0, int(max_ * 2 + 1.0)/2.0, interval))
        ax1.set_yticks(np.arange(int(min_ * 2 - 1.0)/2.0, int(max_ * 2 + 1.0)/2.0, interval/2),
                minor=True)
        ax1.grid(which='major', alpha=0.5)
        ax1.grid(which='minor', alpha=0.2)

    fig.tight_layout()
    fig_p = os.path.join(PLOT_ROOT,  greedy_dir.split('/')[-3] + f'.scale_update.pdf')
    if os.path.exists(fig_p):
        os.rename(fig_p, fig_p.strip('.pdf') + f'.{now}.pdf')
    plt.savefig(fig_p)
    plt.close()

if __name__ == '__main__':
    #check_done()
    run()

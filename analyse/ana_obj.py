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
from analyse.ana_scale_update import target_greedyonline_update_w_scale_update

def load_obj_values():
    EXP_DIR=RESULT_ROOT
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-woodfisher-update_rm_multiple'
    sparsity_obj = {}
    for s in [0.6, 0.8, 0.98]:
        NOWDATE=f'debug_objective_function.test.20211219.17_05_03.scale_update_{s}.arr.txt'
        objs = np.loadtxt(os.path.join(EXP_DIR, EXP_NAME, NOWDATE))
        sparsity_obj[s] = np.array(objs)
    return sparsity_obj

def run():
    now = now_()
    sparsity   = [0.6,  0.8, 0.98]
    scale_update=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_scale     = len(scale_update)
    sparsity_obj = load_obj_values()
    
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6,8)

    ax1 = ax
    color='tab:red'
    ax1.set_xlabel('scaling factors on the weight update')
    ax1.set_ylabel('value of Objective funtions', color=color)
    colors = dict(zip(sparsity, ['r', 'g', 'b']))

    for s in sparsity:
        c = colors[s]
        ax1.plot(scale_update, sparsity_obj[s][:,0], f'{c}*-', label=f'sparsity={s}, only selection')
        ax1.plot(scale_update, sparsity_obj[s][:,1], f'{c}^-', label=f'sparsity={s}, selection + update_rm_multiple_weights')
        ax1.plot(scale_update, sparsity_obj[s][:,2], f'{c}o-', label=f'sparsity={s}, selection + update_rm_single_weight')
   
    ax1.legend(loc='best')
        
    ##ax1.legend(loc="best")
    #interval = max(0.5, int((max_ - min_)/5 * 2.0) / 2.0) 
    #ax1.set_yticks(np.arange(int(min_ * 2 - 1.0)/2.0, int(max_ * 2 + 1.0)/2.0, interval))
    #ax1.set_yticks(np.arange(int(min_ * 2 - 1.0)/2.0, int(max_ * 2 + 1.0)/2.0, interval/2),
    #        minor=True)
    #ax1.grid(which='major', alpha=0.5)
    #ax1.grid(which='minor', alpha=0.2)

    fig.tight_layout()
    fig_p = os.path.join(PLOT_ROOT,  'debug_obj.pdf')
    if os.path.exists(fig_p):
        os.rename(fig_p, fig_p.strip('.pdf') + f'.{now}.pdf')
    plt.savefig(fig_p)
    plt.close()

if __name__ == '__main__':
    run()

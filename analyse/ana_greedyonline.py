import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import shutil
import matplotlib.pyplot as plt
import glob

from common.time import now_
from env import *
from greedy_alg.greedy_io import get_greedy_exp_name
from common.io import _load,_dump, find_float_in_str
from common.time import now_
from analyse.load_results import *

from analyse.cifar10_resnet20_path import *
def load_txt(path):
    data = np.loadtxt(path)
    return data
# collect all results of a certain experiment: the root is named as the script name
def collect_baseline(DIR):
    nowdates = [p for p in glob.glob(os.path.join(DIR, 'log.2022*')) if 'top' not in p ]
    print(DIR)
    for idx, nd in enumerate(nowdates):
        print(nd)
        if idx == 0:
            os.system(f'sh ./collect_baseline_rst_raw.sh {nd} first')
        else:
            os.system(f'sh ./collect_baseline_rst_raw.sh {nd}')
    rst_path = os.path.join(DIR, 'all.top1_top5')
    rst = {}
    if not os.path.exists(rst_path):
        return {}
    with open(rst_path, 'r') as f:
        lines = [l for l in f.readlines() if '.log' in  l]
    for idx, l in enumerate(lines):
        path = l.split(':')[0]
        sp   = find_float_in_str(os.path.basename(path))[0]
        sd   = int(path.split('/')[-2].split('seed')[0][-1])
        acc  = float(l.split(', ')[2]) 
        rst.setdefault(sd, {})
        rst[sd].setdefault(sp, {})
        if 'top1' in l:
            rst[sd][sp]['top1'] = acc
        if 'top5' in l:
            rst[sd][sp]['top5'] = acc

    return rst

def check_pruned(greedy_path):
    data = _load(greedy_path)
    if 'pruned_idx' in data:
        pruned = True
    else:
        pruned = False
    if 'delta_w' in data or 'delta_w.multiple' in data or 'delta_w.single' in data:
        updated = True
    else:
        updated = False
        #import pdb;pdb.set_trace()
    return pruned, updated

############################################################
# check reults
############################################################
def check_done(sparsity, thresholds, ranges, max_no_match, 
                greedy_dir, eval_dir, eval_update_dir):
    # check done
    print('greedy_dir', greedy_dir)
    print('eval_dir', eval_dir)
    print('eval_update_dir', eval_update_dir)
    undone_greedy = []
    undone_eval   = []
    undone_greedy_update = []
    undone_eval_update = []
    for s in sparsity:
        for t in thresholds:
            for r in ranges:
                for m in max_no_match:
                    choice = 'best_iter' 
                    eval_path   = os.path.join(eval_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
                    greedy_path = set_greedy_path(choice, greedy_dir, s,t,r,m)
                    eval_update_path = os.path.join(eval_update_dir, 
                                        get_greedy_exp_name(s,t,r,m) + '.npy')
                    if not os.path.exists(greedy_path):
                        undone_greedy.append(greedy_path)
                    ck_prune, ck_update = check_pruned(greedy_path)
                    if not ck_prune:
                        undone_greedy.append(greedy_path)
                    if not ck_update:
                        undone_greedy_update.append( greedy_path)
                    if not os.path.exists(eval_path):
                        undone_eval.append(eval_path)
                    if not os.path.exists(eval_update_path):
                        undone_eval_update.append(eval_update_path)
                    
                    data_eval = _load(eval_path)
                    if 'top1' not in data_eval[data_eval[choice]]:
                        undone_eval.append(eval_path)
                    data_update_eval = _load(eval_update_path)
                    #if s==0.1 and t==1e-5 and r==10 and m==10:
                    #    import pdb;pdb.set_trace() 
                    if 'top1.updated_weight' not in data_update_eval[data_eval[choice]]:
                        undone_eval_update.append(eval_update_path)
                        #import pdb;pdb.set_trace() 
                        

    print('\n#################todo_greedy')
    print('\n'.join(undone_greedy))
    print('\n#################todo_eval')
    print('\n'.join(undone_eval))
    print('\n#################todo_cal_update_w')
    print('\n'.join(undone_greedy_update))
    print('\n#################todo_eval_update_w')
    print('\n'.join(undone_eval_update))
    #TODO
    #return len(undone_greedy) > 0 or len(undone_eval) > 0 or len(undone_greedy_update) > 0 or len(undone_eval_update) > 0
    return len(undone_greedy) > 0  or len(undone_greedy_update) > 0 or len(undone_eval_update) > 0

def get_best_iter(data, choice):
    if choice in data:
        return data[choice]
    for k1,v1 in data.items():
        if choice in v1:
            return v1[choice]
    return None

def check_greedy_done(s, t, r, m, eval_dir, check_other=True):
    choice = 'best_iter'
    eval_path   = os.path.join(eval_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
    if os.path.exists(eval_path):
        data_eval = _load(eval_path)
        #import pdb;pdb.set_trace()
        rst = data_eval[data_eval[choice]]
        #rst = data_eval[get_best_iter(data_eval, choice)]
        try:
            tmp = rst['top1'][2]
        except:
            pass
        return True 
    # is_greedyed = False: merge other results if there are.
    if check_other:
        return merge_greedy_from_other_run(s, t, r, m, eval_dir)
    else:
        return False
############################################################
# merge reults: greedy
#   only when check_greedy_done(s, t, r, m, eval_dir) = False
#   find other expierments run at other times
#   eval_dir: 
#       main_dir=./prob_regressor_results/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers/results.train_loss_all_samples.20220112.03_57_26/cifarnet_cifar10_1000samples_1000batches_0seed.eval
############################################################
def merge_greedy_from_other_run(s, t, r, m, eval_dir):
    if eval_dir.endswith('/'):
        eval_dir = eval_dir[:-1]
    adir, arch = os.path.dirname(eval_dir), os.path.basename(eval_dir).strip(".eval")
    root, name = os.path.dirname(adir), os.path.basename(adir)
    greedy_dir     = eval_dir.strip('.eval')
    # main_name: results.train_loss_all_samples.20220112.03_57_26
    assert('test' not in name, f"eval_dir should not for test exp: {eval_dir}")
    other_runs = [ p
            for p in glob.glob(os.path.join(root, "results.train_loss_all_samples.*", arch + '.eval')) 
                if 'test' not in p and p != eval_dir]
    for run in other_runs:
        if check_greedy_done(s, t, r, m, run, check_other=False):
            print("Found other greedy results", run)
            # rm the existing greedy_paths
            old_greedy   = glob.glob(os.path.join(greedy_dir, get_greedy_exp_name(s,t,r,m) + '*.npy'))
            now = now_() 
            #import pdb;pdb.set_trace()
            for gp in old_greedy:
                #os.remove(gp)
                os.rename(gp, gp + '_' + now)
            # cp the new greedy_paths
            new_greedy   = glob.glob(os.path.join(run.strip('.eval'), get_greedy_exp_name(s,t,r,m) + '*.npy'))
            if not os.path.exists(greedy_dir):
                os.makedirs(greedy_dir)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            for gp in new_greedy:
                shutil.copy(gp, greedy_dir)
            # cp the new eval_path
            new_eval    = os.path.join(run, get_greedy_exp_name(s,t,r,m) + '.npy')
            shutil.copy(new_eval, eval_dir)
            print(f"Copied greedy results from {os.path.basename(os.path.dirname(run))} to {os.path.basename(eval_dir)}") 
            return True
    return False

############################################################
# collect reults: greedy 
############################################################
def get_eval_greedy(sparsity, thresholds, ranges, max_no_match, eval_dir):
    # get evaluations
    rst_dict = {}
    rst_arr = []
    for s in sparsity:
        rst_dict[s]= {}
        for t in thresholds:
            rst_dict[s][t] = {}
            for r in ranges:
                rst_dict[s][t][r] = {}
                for m in max_no_match:
                    choice = 'best_iter' 
                    eval_path   = os.path.join(eval_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
                    if os.path.exists(eval_path):
                        data_eval = _load(eval_path)
                        rst = data_eval[data_eval[choice]]
                        best_iter = rst['best_iter']
                        if best_iter == 'magnitude':
                            i_best = 0
                        else:
                            i_best = int(best_iter.strip('iter_'))
                        #top1, top5, time
                        top1_top5_itr = [rst['top1'][2], rst['top5'][2], rst['best_iter'], rst['sample_losses'][i_best]]
                    else:
                        top1_top5_itr = ['', '', '', '']
                    arr = [eval_path, s, t, r, m]
                    arr.extend(top1_top5_itr)
                    
                    rst_arr.append(arr)
                    rst_dict[s][t][r][m] = top1_top5_itr
    rst_arr = np.array(rst_arr)
    return rst_dict, rst_arr 
   
def check_update_done(s, t, r, m, eval_dir, eval_update_dir, check_other=True):
    choice = 'best_iter' 
    eval_path   = os.path.join(eval_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
    eval_update_path = os.path.join(eval_update_dir, 
                        get_greedy_exp_name(s,t,r,m) + '.npy')
    #if s==0.4 and t==1e-3 and r==100 and t==20:
    #    import pdb;pdb.set_trace()
    if os.path.exists(eval_path) and os.path.exists(eval_update_path): 
        data_eval = _load(eval_path)
        itr = data_eval[choice]
        data_update_eval = _load(eval_update_path)
        rst = data_update_eval[itr]
        try:
            tmp = rst['top1.updated_weight'][2]
        except:
            pass
        return True
    if check_other:
        return merge_update_from_other_run(s, t, r, m, eval_dir, eval_update_dir)
    else:
        return False

def merge_update_from_other_run(s, t, r, m, eval_dir, eval_update_dir):
    if eval_update_dir.endswith('/'):
        eval_update_dir = eval_update_dir[:-1]
    adir, arch = os.path.dirname(eval_update_dir), os.path.basename(eval_update_dir).strip(".eval")
    root, name = os.path.dirname(adir), os.path.basename(adir)
    arr = name.split('.')
    arr[3], arr[4] = "*", "*"
    other_runs = [ p
            for p in glob.glob(os.path.join(root, ".".join(arr), arch + '.eval')) 
                if 'test' not in p and p != eval_update_dir]
    for run in other_runs:
        if check_update_done(s, t, r, m, eval_dir, run, check_other=False):
            print(f'Other update_dir are found: s={s}, t={t}, r={r}, m={m}', run)
            new_update = os.path.join(run, get_greedy_exp_name(s,t,r,m) + '.npy')
            old_update = os.path.join(eval_update_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
            if not os.path.exists(eval_update_dir):
                os.makedirs(eval_update_dir)
            shutil.copyfile(new_update, old_update)
            return True
    return False
    

############################################################
# collect reults: weight update 
############################################################
def get_eval_update(sparsity, thresholds, ranges, max_no_match, eval_dir, eval_update_dir):
    # get evaluations
    rst_dict = {}
    rst_arr  = []
    for s in sparsity:
        rst_dict[s] = {}
        for t in thresholds:
            rst_dict[s][t] = {}
            for r in ranges:
                rst_dict[s][t][r] = {}
                for m in max_no_match:
                    choice = 'best_iter' 
                    eval_path   = os.path.join(eval_dir, get_greedy_exp_name(s,t,r,m) + '.npy')
                    eval_update_path = os.path.join(eval_update_dir, 
                                        get_greedy_exp_name(s,t,r,m) + '.npy')
                    if os.path.exists(eval_path) and os.path.exists(eval_update_path): 
                        data_eval = _load(eval_path)
                        itr = data_eval[choice]
                        data_update_eval = _load(eval_update_path)
                        rst = data_update_eval[itr]
                        top1_top5 = [rst['top1.updated_weight'][2], rst['top5.updated_weight'][2]]
                    else:
                        #import pdb;pdb.set_trace()
                        top1_top5 = ['', '']
                    arr = [eval_update_path, s, t, r, m]
                    arr.extend(top1_top5)
                    rst_arr.append(arr)
                    rst_dict[s][t][r][m] = top1_top5
    rst_arr = np.array(rst_arr)
    return rst_dict, rst_arr  

def run():
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    #thresholds = [1e-3, 1e-4, 1e-5]
    #ranges     = [10, 100]
    #max_no_match=[10, 20]
    #sparsity   = [0.8]
    thresholds = [1e-5]
    ranges     = [10]
    max_no_match=[10]

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
   
    # plots for greedy parameters
    fig, ax = plt.subplots(len(sparsity),1)
    # Set the ticks and ticklabels for all axes
    plt.setp(ax, xticks=range(len(thresholds)), xticklabels=thresholds,
            xlabel='thresholds', ylabel='test_acc (%)')
    # Use the pyplot interface to change just one subplot...
    #plt.sca(axes[1, 1])
    #plt.xticks(range(3), ['A', 'Big', 'Cat'], color='red')

    # Draw figures for 
    color='tab:red'
    for idx,s in enumerate(sparsity):
        for r in ranges:
            for m in max_no_match:
                color = eval_colors[f'r{r}_m{m}']
                ax[idx].set_title(f'sparisty={s}')

                # plots for init_method=mag
                xticks = np.arange(len(thresholds))
                acc, acc_update = all_select_mag[s][r][m], all_update_mag[s][r][m]
                if idx == 0:
                    ax[idx].plot(xticks, acc, f'{color}o-', 
                        label=f'r{r}-m{m}-init_mag-no_update_w')
                    ax[idx].plot(xticks, acc_update, f'{color}^-', 
                        label=f'r{r}-m{m}-init_mag-update_w')
                else:
                    ax[idx].plot(xticks, acc, f'{color}o-') 
                    ax[idx].plot(xticks, acc_update, f'{color}^-') 

                # plots for init_method=woodfisherblock
                acc, acc_update = all_select_wfb[s][r][m], all_update_wfb[s][r][m]
                if idx == 0:
                    ax[idx].plot(xticks, acc, f'{color}o--', 
                        label=f'r{r}-m{m}-init_woodfisherblock-no_update_w')
                    ax[idx].plot(xticks, acc_update, f'{color}^--', 
                        label=f'r{r}-m{m}-init_woodfisherblock-update_w')
                else:
                    ax[idx].plot(xticks, acc, f'{color}o--') 
                    ax[idx].plot(xticks, acc_update, f'{color}^--')

    fig.legend(loc='center right')
    fig.tight_layout()
    fig.set_size_inches(10, 50)
    now = now_()
    fig_p = os.path.join(PLOT_ROOT,  greedy_dir.split('/')[-3] + f'.greedy.pdf')
    if os.path.exists(fig_p):
        os.rename(fig_p, fig_p.strip('.pdf') + f'.{now}.pdf')
    plt.savefig(fig_p)
    plt.close()

    fig, ax = plt.subplots(3)
    fig.set_size_inches(8, 16)

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

    ax1.legend(loc='best')
    
    
    ax1 = ax[1]
    color='tab:red'
    ax1.set_xlabel('Target sparsity level')
    ax1.set_ylabel('test_acc (%)', color=color)
    
    # plot for baselines
    ax1.plot(sparsity, [unpruned_acc[s] for s in sparsity], 'k*-', label='uncompressed model')
    ax1.plot(sparsity, mag_rst, 'bo-.', label='magnitude')
    ax1.plot(sparsity, wfb_no_update_rst, 'go-.', label='woodfisherblock_no_update_w')
   
    # plot for init_method=mag
    ax1.plot(sparsity, [best_select_mag[s] for s in sparsity], 'c^-.', 
                label = 'greedy-init_mag-no_update_w')
    # plot for init_method=woodfisherblock
    ax1.plot(sparsity, [best_select_wfb[s] for s in sparsity], '^-.', color='tab:blue',
                label = 'greedy-init_woodfisherblock-no_update_w')
    ax1.legend(loc='best')

    ax1 = ax[2]
    color='tab:red'
    ax1.set_xlabel('Target sparsity level')
    ax1.set_ylabel('test_acc (%)', color=color)
    
    # plot for baselines
    ax1.plot(sparsity, [unpruned_acc[s] for s in sparsity], 'k*-', label='uncompressed model')
    ax1.plot(sparsity, mag_rst, 'bo-.', label='magnitude')
    ax1.plot(sparsity, wfb_rst, 'ro-', label='woodfisherblock')
   
    # plot for init_method=mag
    ax1.plot(sparsity, [best_update_mag[s] for s in sparsity], 'm^-', 
                label = 'greedy-init_mag-update_w')
    # plot for init_method=woodfisherblock
    ax1.plot(sparsity, [best_update_wfb[s] for s in sparsity], '^-', color='tab:pink',
                label = 'greedy-init_woodfisherblock-update_w')
    ax1.legend(loc='best')
    
    fig.tight_layout()
    fig_p = os.path.join(PLOT_ROOT,  greedy_dir.split('/')[-3] + f'.pdf')
    if os.path.exists(fig_p):
        os.rename(fig_p, fig_p.strip('.pdf') + f'.{now}.pdf')
    plt.savefig(fig_p)
    plt.close()

if __name__ == '__main__':
    #check_done()
    run()

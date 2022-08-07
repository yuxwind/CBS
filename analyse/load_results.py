import os,sys
sys.path.append('..')
import numpy as np

from common.time import now_
from common.io import _load, _dump
from env import ROOT, RESULT_ROOT
from greedy_alg.greedy_io import get_greedy_exp_paths, get_greedy_exp_name, get_greedy_exp_name_with_mag

def get_greedy_path_last_iter(greedy_dir, s,t,r,m):
    # NOTE: get_greedy_exp_paths() has sorted exps in the order they created
    return get_greedy_exp_paths(greedy_dir,s,t,r,m, 
                    only_last_iter=False, include_mag=True)[-1]
     
def get_greedy_path_mag(greedy_dir, s,t,r,m):
    return os.path.join(greedy_dir, get_greedy_exp_name_with_mag(s,t,r,m))

def get_greedy_path_best_iter(greedy_dir,s,t,r, m):
    eval_dir   = greedy_dir + '.eval'
    exp_name   = get_greedy_exp_name(s,t,r,m)
    eval_path  = os.path.join(eval_dir, exp_name + '.npy')
    data       = _load(eval_path)
    best_iter  = None
    if 'best_iter' in data:
        print(data['best_iter'])
        best_iter_i = data['best_iter']
        if type(best_iter_i) == str:
            best_iter = best_iter_i
        else:
            best_iter   = 'magnitude' if best_iter_i == 0 else f'iter_{best_iter_i}'
            data['best_iter'] = best_iter
    else:
        for k,v in data.items():
            if 'best_iter' in v:
                data['best_iter'] = k
                best_iter = k.strip('-') 
                break
        if best_iter is not None:
            data['best_iter'] = best_iter
            _dump(eval_path, data)
    if best_iter is not None: 
        if '-' not in best_iter:
            best_iter = '-' + best_iter 
        greedy_path   = os.path.join(greedy_dir, exp_name + best_iter + '.npy')
    else:
        greedy_path   = get_greedy_path_last_iter(greedy_dir, s,t,r,m)
    return greedy_path
    
def set_greedy_path(target, greedy_dir,s,t,r,m):
    assert(target in ['last_iter', 'mag', 'best_iter'], 'target is unknown')
    if target == 'best_iter': 
        greedy_path = get_greedy_path_best_iter(greedy_dir,s,t,r, m)
    elif target == 'mag':
        greedy_path =  get_greedy_path_mag(greedy_dir,s,t,r,m)
    else:
        greedy_path = get_greedy_path_last_iter(greedy_dir,s,t,r,m)
    
    return greedy_path

def is_evaluated(greedy_path, eval_with_weight_update=False):
    eval_dir  = os.path.dirname(greedy_path) + '.eval'
    eval_name = '-'.join(os.path.basename(greedy_path).split('-')[:-1]) + '.npy'
    try:
        eval_path = os.path.join(eval_dir, eval_name) 
        data      = _load(eval_path)
    except:
        import pdb;pdb.set_trace()

    itr       = os.path.basename(greedy_path).split('-')[-1].split('.')[0]
    if itr == 'magnitude':
        itr = '-' + itr
    if itr not in data:
        return False, None, None

    if eval_with_weight_update and 'top1' in data[itr]:
        return True, data[itr]['top1'], data[itr]['top5']
    if not eval_with_weight_update and 'top1.updated_weight' in data[itr]:
        return True, data[itr]['top1.updated_weight'], data[itr]['top5.updated_weight']
    
    return False, None, None

def get_scales(methodToRun):
    # extract the scales from the paths of results
    GREEDY_DIR, EVAL_DIR, EVAL_update_DIR = methodToRun(0)
    pattern = os.path.dirname(EVAL_update_DIR).split('scale_update_')[0] + 'scale_update_*'
    scales  = sorted([float(d.split('scale_update_')[1]) for d in glob(pattern)])
    return scales

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
        greedy_dir, eval_dir, eval_update_single_dir, eval_update_multiple_dir= methodToRun(scale)
        #undone = check_done(
        #                sparsity, thresholds, ranges, max_no_match,
        #                greedy_dir, eval_dir, eval_update_dir)
        #if undone:
        #    return
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

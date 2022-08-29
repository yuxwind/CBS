import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
#sys.path.append('..')
import pathlib

#import pdb;pdb.set_trace()

import numpy as np
from glob import glob

from common.io import find_float_in_str, _load, _dump
from env import ROOT
from env_cfg import greedy_cls, eval_root 


def write_greedy_results(args, results_top1, results_top5, updated_weight=False, extra_dict={}):
    greedy_dir = args.greedy_dir
    # Set args.greedy_path for weight update exp only
    if args.greedy_path is None:
        exp = get_greedy_exp_name(args.target_sparsity, args.threshold, args.range,  args.max_no_match)
        if type(extra_dict['best_iter']) == str:
            itr = extra_dict['best_iter']
        else:
            if extra_dict['best_iter'] == 0:
                itr = 'magnitude'
            else:
                itr = f"iter_{extra_dict['best_iter']}"
        greedy_path = os.path.join(greedy_dir, exp + '-' + itr + '.npy') 
    else:
        # the greedy exp should go here
        _, fname = os.path.dirname(args.greedy_path), os.path.basename(args.greedy_path)
        cfgs = fname.split('-')
        exp  = '-'.join(cfgs[:2])
        itr = cfgs[2].split('.')[0]
        greedy_path = args.greedy_path

    eval_dir = greedy_dir + '.eval'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    acc_tag = '.updated_weight' if updated_weight else ''
    
    # load evaluate results if it exists
    evals = {}
    eval_path = os.path.join(eval_dir, exp + '.npy')
    if os.path.exists(eval_path):
        evals = _load(eval_path)
    # load greedy results:
    greedy_rst = _load(greedy_path)

    # update the results 
    evals.setdefault(itr, {})
    evals[itr].update({'iter_time': greedy_rst['iter_time'], 
                    'accumulated_time': greedy_rst['accumulated_time'],
                    f'top1{acc_tag}': results_top1,
                    f'top5{acc_tag}': results_top5,
                    })
    evals[itr].update(extra_dict)
    if args.greedy_path is None:
        evals['best_iter'] = itr
    #evals['best_iter'] = itr

    _dump(eval_path, evals)
    print("\n")
    print('##################################################################')
    print(f'Saved the greedy results (weight masks) at gredy_path: \n\t{greedy_path}')
    print(f'Saved the evaluation results at eval_path: \n\t{eval_path}')
    print('##################################################################')
    print('\n')

def extract_greedy_cfg(greedy_path):
    name = os.path.basename(greedy_path)
    sparsity, T, r, m = find_float_in_str(name)[:4]
    r = int(r)
    m = int(m)
    return sparsity, T, r,m 

def extract_greedy_exp_cfgs(greedy_dir):
    exps = [os.path.basename(p) for p in glob(os.path.join(greedy_dir, '*.npy'))
                if 'magnitude' not in p]
    cfgs = np.stack([find_float_in_str(p)[:4] for p in exps], axis=0)
    sparsity   = sorted(set(cfgs[:,0]))
    thresholds = sorted(set(cfgs[:,1]))
    ranges     = sorted(set(cfgs[:,2].astype(int)))
    max_no_matches = sorted(set(cfgs[:,3].astype(int)))

    print('sparsity:', sparsity)
    print('thresholds', thresholds)
    print('ranges', ranges)
    print('max_no_matches', max_no_matches)
    return sparsity,thresholds,ranges,max_no_matches

def get_greedy_exp_name(sparsity, threshold, arange,  max_no_match):
    return f'sparsity{sparsity:.2f}_T{threshold:.2e}_Range{arange:d}_NOMATCH{max_no_match}'

def get_greedy_exp_name_with_mag(sparsity, threshold, arange,  max_no_match):
    return get_greedy_exp_name(sparsity, threshold, arange,  max_no_match) + '-magnitude.npy'

def get_greedy_exp_name_with_itr(sparsity, threshold, arange,  max_no_match, itr):
    return get_greedy_exp_name(sparsity, threshold, arange,  max_no_match) + f'-iter_{itr}.npy'

def get_iters_of_exp(greedy_dir, sparsity, threshold, arange,  max_no_match):
    exp = get_greedy_exp_name(sparsity, threshold, arange,  max_no_match)
    exp_paths = [f for f in glob(os.path.join(greedy_dir, exp+'*.npy')) if 'iter_' in f]
    iters = sorted([int(find_float_in_str(os.path.basename(f))[-1]) for f in exp_paths])
    return iters

def get_pruned_info(exp_root, sparsity, threshold, arange,  max_no_match, itr, attr='pruned_idx'):
    if attr == 'pruned_idx':
        if itr == 'magnitude' or 'mag':
            exp_path = get_greedy_exp_name_with_mag(sparsity, threshold, arange,  max_no_match)
        else:
            exp_path = get_greedy_exp_name_with_itr(sparsity, threshold, arange,  max_no_match, itr)
        
        exp_path = os.path.join(ROOT, f'prob_regressor_results/{greedy_cls}', exp_root, exp_path)
        return _load(exp_path).item['pruned_idx']
    else:
        name = get_greedy_exp_name(sparsity, threshold, arange,  max_no_match)
        exp_path = os.path.join(ROOT, f'prob_regressor_results/{greedy_cls}', exp_root + '.eval', name + '.npy')
        return exp_path[f'iter_{itr}'][attr] 

def get_train_loss(greedy_dir, sparsity, threshold, arange,  max_no_match, 
        include_mag=True):
    iters = get_iters_of_exp(greedy_dir, sparsity, threshold, arange,  max_no_match)
    name  = get_greedy_exp_name(sparsity, threshold, arange,  max_no_match)
    exp_path = os.path.join(greedy_dir + '.eval', name + '.npy')
    if not os.path.exists(exp_path):
        print(f'!!! path not exist: {exp_path}')
        return None
    data  = _load(exp_path)
    key   = 'after_prune_loss_train'
    s_iter= ['magnitude'] if include_mag else []
    s_iter.extend([f'iter_{it}' for it in iters])
    acc_of_all_iters = []

    for it in s_iter:
        if it in data:
            acc_of_all_iters.append(data[it][key])
        else:
            print(f'{it} not in {exp_path}')
            acc_of_all_iters = None
            break
    return acc_of_all_iters

def get_pruned_idx_for_all_iters(greedy_dir, sparsity, threshold, arange,  max_no_match, 
        include_mag=True):
    exp_paths = get_greedy_exp_paths(greedy_dir, sparsity, threshold, arange,  max_no_match, 
            only_last_iter=False, include_mag=include_mag)
    pruned_idx = [_load(f)['pruned_idx'] for f in exp_paths]
    return pruned_idx

def get_greedy_for_all_iters(greedy_dir, sparsity, threshold, arange,  max_no_match, 
        include_mag=True):
    exp_paths = get_greedy_exp_paths(greedy_dir, sparsity, threshold, arange,  max_no_match, 
            only_last_iter=False, include_mag=include_mag)
    pruned_idx = []
    obj = []
    #meta_keys = Meta.set().names
    for f in exp_paths:
        data = _load(f)
        pruned_idx.append(data['pruned_idx'])
        if 'iter_1' in f and include_mag:
            obj.append(data['meta']['obj_before_swap'][0])
        if 'iter' in f:
            obj.append(data['meta']['obj_after_swap'][-1])
    return pruned_idx, obj

def get_pruned_acc_for_all_iters(greedy_dir, sparsity, threshold, arange, max_no_match, is_top1=True,
        include_mag=True):
    iters = get_iters_of_exp(greedy_dir, sparsity, threshold, arange,  max_no_match)
    name  = get_greedy_exp_name(sparsity, threshold, arange,  max_no_match)
    exp_path = os.path.join(greedy_dir + '.eval', name + '.npy')
    if not os.path.exists(exp_path):
        print(f'!!! path not exist: {exp_path}')
        return None
    data  = _load(exp_path)
    key   = 'top1' if is_top1 else 'top5'
    s_iter= ['magnitude'] if include_mag else []
    s_iter.extend([f'iter_{it}' for it in iters])
    acc_of_all_iters = []

    for it in s_iter:
        if it in data:
            #acc_of_all_iters.append(data[it][key][2])
            acc_of_all_iters.append(data[it])
        else:
            print(f'{it} not in {exp_path}')
            acc_of_all_itere = None
            break
    # keys: ['iter_time', 'accumulated_time', 'top1', 'top5', 'after_prune_loss_train', 'after_prune_acc_top1_train', 'after_prune_acc_top5_train']
    top1_test  = [it['top1'][2] for it in acc_of_all_iters]
    top1_train = [it['after_prune_acc_top1_train'] for it in acc_of_all_iters]
    loss_train = [it['after_prune_loss_train'] for it in acc_of_all_iters]
    return top1_test, top1_train, loss_train
       
def get_greedy_exp_paths(greedy_dir, sparsity, threshold, arange,  max_no_match, 
        only_last_iter=True, include_mag=False):
    exp = get_greedy_exp_name(sparsity, threshold, arange,  max_no_match)
    if include_mag:
        exp_paths = [os.path.join(greedy_dir, exp+'-magnitude.npy')]
    else:
        exp_paths = []
    itrs = get_iters_of_exp(greedy_dir, sparsity, threshold, arange,  max_no_match)
    exp_paths.extend([os.path.join(greedy_dir, exp+f'-iter_{it}.npy') for it in itrs])
    if len(exp_paths) == 0:
        return None
    if only_last_iter:
        last_iter = itrs[-1]
        exp_path = os.path.join(greedy_dir, exp + f'-iter_{int(last_iter)}.npy')
        return exp_path
    else:
        return exp_paths

def get_greedy_all_exps(greedy_dir, only_last_iter=True):
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_dir)
    exp_paths = []
    for s in sparsity:
        for m in max_no_matches:
            for r in ranges:
                for t in thresholds:
                    paths = get_greedy_exp_paths(greedy_dir, s, t, r,  m, 
                            only_last_iter=only_last_iter)
                    if paths is not None: 
                        if only_last_iter:
                            paths = [paths]
                        exp_paths.append(paths)
    return exp_paths

def collect_eval(greedy_dir, collect_greedy=True, collect_w_adjustment=False):
    sparsity,thresholds,ranges,max_no_matches = extract_greedy_exp_cfgs(greedy_dir)
    # debug
    #sparsity = np.append(np.arange(0.1,0.9,0.1), 0.85)
    #thresholds = [1e-4, 1e-05]
    #ranges = [10, 100]
    #max_no_matches = [20]

    s_top1 = ''
    s_time = ''
    s_top1_update = ''
    s_time_update = ''
    top1_change = []
    for s in sparsity:
        top1_change = []
        for m in max_no_matches:
            info = f'sparsity={s}, max_no_matches={m}\n'
            #info = '\n'
            s_top1 += info
            s_time += info
            s_top1_update += info
            s_time_update += info
            for r in ranges:
                for t in thresholds:
                    name = f'sparsity{s}_T{t:.2e}_Range{r}_NOMATCH{m}.npy'
                    path = os.path.join(greedy_dir, name)
                    if not os.path.exists(path):
                        s_top1 += 'nan \t'
                        s_time += 'nan \t'
                        s_top1_update += 'nan \t'
                        s_time_update += 'nan \t'
                        continue
                    data = np.load(path, allow_pickle=True)
                    #import pdb;pdb.set_trace()
                    if len([k for k in data.keys() if 'iter' in k]) == 0:
                        s_top1 += 'nan \t'
                        s_time += 'nan \t'
                        s_top1_update += 'nan \t'
                        s_time_update += 'nan \t'
                        continue
                    data = np.load(path, allow_pickle=True)

                    final_iter = max([int(k.split('_')[1]) for k in data.keys() if 'iter' in k])
                    itr_data = data[f'iter_{final_iter}']
                    #import pdb;pdb.set_trace()
                    if collect_greedy: 
                        top1 = data[f'iter_{final_iter}']['top1'][-2]
                        time = data[f'iter_{final_iter}']['accumulated_time']
                        s_top1 += f'{top1}\t'
                        s_time += f'{time:.2f}\t'
                    if collect_w_adjustment:
                        if 'top1.updated_weight' in data[f'iter_{final_iter}']:
                            top1_update = data[f'iter_{final_iter}']['top1.updated_weight'][-2]
                        else:
                            top1_update = np.inf
                        if 'cal_delta_time' in data[f'iter_{final_iter}']:
                            time_update = data[f'iter_{final_iter}']['cal_delta_time']
                        else:
                            time_update = np.inf
                        s_top1_update += f'{top1_update - top1:.2f}\t'
                        top1_change.append(top1_update - top1)
                        #s_time_update += f'{time_update:.2f}\t'
                        s_top1 += f'{top1_update}\t'
                        s_time += f'{time_update:.2f}\t'
                s_top1 += '\n'
                s_time += '\n'
                s_top1_update += '\n'
                s_time_update += '\n'
        #print(s, np.array(top1_change).mean(), np.std(np.array(top1_change)))
    print('#####TOP1:')
    print(s_top1)
    print('#####time:')
    print(s_time)
    if collect_w_adjustment:
        print('#####TOP1_update_weights:')
        #print(s_top1_update)
    #print('#####time_update_weights:')
    #print(s_time_update)

def save_idx_2_module(idx_2_module_path, idx_2_module):
    if not os.path.exists(idx_2_module_path):
        np.save(idx_2_module_path, idx_2_module)

def load_idx_2_module(fpath):
    data = _load(fpath)
    return data

def get_module(idx_2_module, idx):
    for k,v in idx_2_module.items():
        if idx >= k[0] and idx < k[1]:
            return v
    return 'Not Found'

def get_modules(idx_2_module, idxs):
    m = []
    for i in idxs:
        m.append(get_module(idx_2_module, i))
    return m

def is_fc(idx_2_module, idx):
    return 'fc' in get_module(idx_2_module, idx)

def in_same_block(idx_2_module, idx1, idx2):
    m1 = get_module(idx_2_module, idx1)
    m2 = get_module(idx_2_module, idx2)
    #return m1.split('.')[1] == m2.split('.')[1]
    return m1 == m2

if __name__ == '__main__':
    #eval_root = os.path.join(ROOT, 'prob_regressor_results/greedy',
    #        #'resnet20_cifar10_1000samples_1000batches_0seed_allweights.eval')
    #        #'resnet20_cifar10_3000samples_3000batches_0seed.eval')
    #        #'mlp_mnist_10000samples_10000batches.new.eval')
    #        'resnet20_cifar10_1000samples_1000batches_0seed.eval')
    collect_eval(eval_root)
   
    # Debug without vs with decoposeH 
    #eval_root = os.path.join(ROOT, 'prob_regressor_results/greedy',
    #        'greedy_v1_results_without_decomposeH',
    #        'mlp_mnist_10000samples_10000batches.new.eval')
    #print('without decomposeH + old eval==============')
    #collect_eval(eval_root)
    #
    #eval_root = os.path.join(ROOT, 'prob_regressor_results/greedy',
    #        'mlp_mnist_10000samples_10000batches.new.eval')
    #print('without decomposeH + new eval==============')
    #collect_eval(eval_root)
    #
    #eval_root = os.path.join(ROOT, 'prob_regressor_results/greedy',
    #        'greedy_v1_results_decomposeH',
    #        'mlp_mnist_10000samples_10000batches.new.eval')
    #print('decomposeH + new eval==============')
    #collect_eval(eval_root)

import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))


import numpy as np
import shutil

from common.time import now_
from analyse.load_results import *
from env import *

script = sys.argv[1]
GPUs = [s.strip() for s in  sys.argv[2].split(',')]

NOW      = now_("%Y%m%d.%H_%M_%S")
if len(sys.argv) == 4:
    NOW  = 'test'
    
ROOT_DIR = './scripts/run_batch.' + NOW # default: %Y%m%d.%H_%M_%S

PARAL_CNT= 1
#GPUs     = [1]
print(f'NOW: {NOW}')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

arch = 'mobilenet'
greedy_option = 'one' # for mag + update 
scale_option  = 'one' # run the update once and save the upate (delta_w)

def get_hyperparams():
    assert(arch in ['mlp', 'cifarnet', 'resnet20', 'resnet50', 'mobilenet'], f"arch {arch} not in ['mlp', 'cifarnet', 'resnet20', 'resnet50']!!!!!!!!!!")
    assert(greedy_option in ['all', 'one', 'manual'], f"greedy_option {greedy_option} not in ['all', 'test', 'manual']!!!!!!!!!!!!!")
    assert(scale_option in ['all', 'one'], f"greedy_option {greedy_option} not in ['all', 'test', 'manual']!!!!!!!!!!!!!")
    # ResNet20
    sparsity_resnet20   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    # CifarNet
    sparsity_cifarnet   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    sparsity_cifarnet.extend([0.75, 0.857, 0.875, 0.933, 0.95, 0.967, 0.98, 0.986, 0.99]) # additional extreme sparsity for CIFARNet
    # Resnet50 and MobileNet
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    

    if greedy_option == 'all':
        thresholds = [1e-3, 1e-4, 1e-5]
        ranges     = [10, 100]
        max_no_match=[10, 20]
    elif greedy_option == 'one': 
        thresholds = [1e-5]
        ranges     = [10]
        max_no_match=[10]
        sparsity   = [0.6]

    if scale_option == 'all':
        #scale_update=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scale_update=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else: 
        scale_update=[1.0]
    return sparsity, thresholds, ranges, max_no_match, scale_update

def run_strm_s(sparsity, thresholds, ranges, max_no_match, scale_update, paral_cnt=1):
    cmds = []
    for s in sparsity:
        for t in thresholds:
            for r in ranges:
                for m in max_no_match:
                    for u in scale_update:
                        #cmds.append(f'bash {script} {s} {t:.2e} {r} {m} {u}')
                        cmds.append(f'bash {script} {s} {t:.2e} {r} {m} {u}')
    idx = int(paral_cnt * len(GPUs))
    for i in range(idx):
        if PARAL_CNT >= 1:
            GPU    = GPUs[i % len(GPUs)]
        else:
            gpu_per_job = int(1/PARAL_CNT)
            gpu         = GPUs[i*gpu_per_job:max(len(GPUs), (i+1)*gpu_per_job)]
            GPU    = ','.join([str(g) for g in gpu])
            #import pdb;pdb.set_trace()
        f_name = f'./{ROOT_DIR}/script{i}.sh'
        cmds_i = [cmds[j] + f' {GPU} {NOW}' for j in range(len(cmds)) if j % idx == i] 
        cmds_s = '\n'.join(cmds_i)
        with open(f_name, 'w') as f:
            f.write(cmds_s)
        if NOW == 'test':
            os.system(f'sh {f_name}')
        else:
            os.system(f'sh {f_name} > ./{ROOT_DIR}/script{i}.log 2>&1 &')
            print(f'log: ./{ROOT_DIR}/script{i}.log')

if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    sparsity, thresholds, ranges, max_no_match, scale_update = get_hyperparams()
    run_strm_s(sparsity, thresholds, ranges, max_no_match, scale_update, paral_cnt=PARAL_CNT)

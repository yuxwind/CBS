import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import shutil

from common.time import now_
from analyse.load_results import *
from env import *

script = sys.argv[1]
NOW      = now_("%Y%m%d.%H_%M")
if len(sys.argv) == 3:
    NOW  = 'test'
    
ROOT_DIR = './scripts/run_batch.' + NOW # default: %Y%m%d.%H_%M_%S

#PARAL_CNT= 3
#GPUs     = [0,1,2]
PARAL_CNT= 1
GPUs     = [0]

print(f'NOW: {NOW}')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

def target_greedyonline():
    EXP_DIR=RESULT_ROOT
    #EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers'
    #NOWDATE='train_loss_all_samples.20211202.22_52'
    EXP_NAME='cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers'
    NOWDATE='train_loss_all_samples.20211202.22_59'
    ARCH_NAME="resnet20_cifar10_1000samples_1000batches_0seed"
    GREEDY_DIR=f"{EXP_DIR}/{EXP_NAME}/results.{NOWDATE}/{ARCH_NAME}"
    return GREEDY_DIR

def run(paral_cnt=1):
    #sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    #thresholds = [1e-3, 1e-4, 1e-5]
    #ranges     = [10, 100]
    #max_no_match=[10, 20]
    sparsity   = [0.1]
    thresholds = [1e-4]
    ranges     = [10]
    max_no_match=[20]

    cmds = []
    greedy_dir = target_greedyonline() 
    for s in sparsity:
        for t in thresholds:
            for r in ranges:
                for m in max_no_match:
                    greedy_path = set_greedy_path('best_iter', greedy_dir, s,t,r,m)
                    #greedy_path = set_greedy_path('mag', greedy_dir, s,t,r,m)
                    print(greedy_path)
                    #is_evaludated, top1, top5 = is_evaluated(greedy_path, eval_with_weight_update=True)
                    is_evaludated = False
                    if is_evaludated:
                        print(greedy_path, top1, top5)
                    else:
                        cmds.append(f'bash {script} {s} {t:.2e} {r} {m} {greedy_path}')
    idx = paral_cnt * len(GPUs)
    for i in range(idx):
        GPU    = GPUs[i % len(GPUs)]
        f_name = f'./{ROOT_DIR}/script{i}.sh'
        cmds_i = [cmds[j] + f' {GPU} {NOW}' for j in range(len(cmds)) if j % idx == i] 
        cmds_s = '\n'.join(cmds_i)
        with open(f_name, 'w') as f:
            f.write(cmds_s)
        if NOW == 'test':
            os.system(f'sh {f_name}')
        else:
            import pdb;pdb.set_trace()
            os.system(f'sh {f_name} > ./{ROOT_DIR}/script{i}.log 2>&1 &')
            print(f'log: ./{ROOT_DIR}/script{i}.log')

if __name__ == '__main__':
    run(paral_cnt=PARAL_CNT)

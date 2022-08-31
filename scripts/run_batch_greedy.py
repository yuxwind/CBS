import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

from common.time import now_
from analyse.ana_greedyonline import check_greedy_done, check_update_done

script = sys.argv[1]
GPUs = [s.strip() for s in  sys.argv[2].split(',')]
NOW      = now_("%Y%m%d.%H_%M_%S")
if len(sys.argv) == 4:
    NOW  = 'test' 
ROOT_DIR = './scripts/run_batch.' + NOW # default: %Y%m%d.%H_%M_%S
#script  = 'scripts/greedyonline_cifar10_resnet20.update_w.sh'
#PARAL_CNT= 6

#script   = 'scripts/greedyonline_cifar10_resnet20.sh'
PARAL_CNT= 1

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

arch = script.split('-')[1]
init = 'mag_perb' if 'magperb' in script  or 'mag_perb' in script else 'mag'

if arch == 'cifarnet':
    from analyse.cifar10_cifarnet_path import target_greedyonline_init_magperb, target_greedyonline_init_mag
if arch == 'resnet20':
    from analyse.cifar10_resnet20_path import target_greedyonline_init_magperb, target_greedyonline_init_mag
if arch == 'mlpnet':
    from analyse.mnist_mlpnet_path import target_greedyonline_init_magperb, target_greedyonline_init_mag
if arch == 'mobilenet':
    from analyse.imagenet_mobilenet_path import target_greedyonline_init_magperb, target_greedyonline_init_mag


if init == 'mag':
    greedy_dir, eval_dir, dir_single_update_fmt, dir_multiple_update_fmt = target_greedyonline_init_mag()
else:
    greedy_dir, eval_dir, dir_single_update_fmt, dir_multiple_update_fmt = target_greedyonline_init_magperb()

if 'multiple' in script:
    dir_update_fmt = dir_multiple_update_fmt
elif 'single' in script:
    dir_update_fmt = dir_single_update_fmt
else:
    dir_update_fmt = 'NONE'
print("##########INFO##########")
print(script)
print(f'arch={arch}, init={init}, dir_update_fmt={dir_update_fmt}')
print(f'greedy_dir: {greedy_dir}')
print(f'eval_dir: {eval_dir}')
print("##########INFO##########")

def run(paral_cnt=1):
    # ResNet20
    resnet20_sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    # CIFARNet
    cifarnet_sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    cifarnet_sparsity.extend([0.75, 0.857, 0.875, 0.933, 0.95, 0.967, 0.98, 0.986, 0.99]) # additional extreme sparsity for CIFARNet 
    # ResNet50 
    resnet50_sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # MobileNet
    #mobilenet_sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mobilenet_sparsity   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # mlpnet
    mlpnet_sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]

    if arch == 'cifarnet':
        sparsity = cifarnet_sparsity
    if arch == 'resnet20':
        sparsity = resnet20_sparsity
    if arch == 'mlpnet':
        sparsity = mlpnet_sparsity
    if arch == 'mobilenet':
        sparsity = mobilenet_sparsity
    #NOTE: this is for running time: use the same sparsity for all experiments
    sparsity = mlpnet_sparsity
    #thresholds = [1e-3, 1e-4, 1e-5]
    #thresholds = [1e-3, 1e-4]
    thresholds = [1e-4]
    #ranges     = [10, 100]
    #max_no_match=[10, 20]
    ranges     = [10]
    #max_no_match=[10, 20]
    max_no_match=[20]
    #thresholds = [2.0, 3.0, 4.0, 6.0, 8.0]
    #seeds = [0, 1, 2, 3, 4]
    seeds = [3,4]
    
    #NOTE: this is for running time: 
    seeds = [6]
    
    sparsity   = [0.95]
    #thresholds = [1e-4]
    #ranges     = [10]
    #max_no_match=[20]
    #seeds = [3]

    print(sparsity)
    cmds = []
    N_exps_all = 0
    for s in sparsity:
        for t in thresholds:
            for r in ranges:
                for m in max_no_match:
                    for sd in seeds: 
                        N_exps_all += 1
                        is_greedyed = check_greedy_done(s, t, r, m, eval_dir%sd)
                        #is_greedyed = False
                        if not is_greedyed:
                            cmds.append(f'bash {script} {s} {t:.2e} {r} {m} {sd}')
                        else:
                            print(f'DONE: {s} {t:.2e} {r} {m} {sd}')
    print(f'####{len(cmds)}/{N_exps_all} to run')
    import pdb;pdb.set_trace()
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
            os.system(f'sh {f_name} > ./{ROOT_DIR}/script{i}.log 2>&1 &')

if __name__ == '__main__':
    run(paral_cnt=PARAL_CNT)

import os,sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

from common.time import now_
from analyse.ana_greedyonline import collect_baseline

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
#GPUs     = [0,1]

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

if 'cifarnet' in script:
    from analyse.cifar10_cifarnet_path import baseline
if 'resnet20' in script:
    from analyse.cifar10_resnet20_path import baseline
if 'mlpnet' in script:
    from analyse.mnist_mlpnet_path import baseline
if 'mobilenet' in script:
    from analyse.imagenet_mobilenet_path import baseline

WFB_DIR, WFB_no_update_DIR = baseline()
if 'no_weight_update' in script:
    DIR = WFB_no_update_DIR
else:
    DIR = WFB_DIR
rst = collect_baseline(DIR)

def run(paral_cnt=1):
    # ImageNet
    #sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Cifarnet
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    sparsity.extend([0.75, 0.857, 0.875, 0.933, 0.95, 0.967, 0.986, 0.99]) # additional extreme sparsity for CifarNet 
    
    # ResNet50
    #sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # MobileNet, ResNet20
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # MNIST
    sparsity   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    #seeds      = [0,1,2,3,4]
    seeds      = [6]

    cmds = []
    for s in sparsity:
        for se in seeds:
            try:
                acc = rst[se][s]
            except:
                cmds.append(f'bash {script} {s} {se}')
    print(f'{len(cmds)}/{len(sparsity) * len(seeds)} exps are to run')
    #import pdb;pdb.set_trace()
    idx = int(paral_cnt * len(GPUs))
    for i in range(idx):
        if PARAL_CNT >= 1:
            GPU    = GPUs[i % len(GPUs)]
        else:
            gpu_per_job = int(1/PARAL_CNT)
            gpu         = GPUs[i*gpu_per_job:max(len(GPUs), (i+1)*gpu_per_job)]
            #import pdb;pdb.set_trace()
            GPU    = ','.join([str(g) for g in gpu])
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

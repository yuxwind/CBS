import os
import sys
from greedy_io import get_greedy_exp_name

sys.path.append('..')
from env_cfg import FISHER_PATH, greedy_cls

#FISHER_PATH='mlp_mnist_10000samples_10000batches.new.pkl'
#FISHER_PATH='resnet20_cifar10_1000samples_1000batches_0seed.pkl'
#FISHER_PATH='resnet20_cifar10_3000samples_3000batches_0seed.pkl'
#FISHER_PATH='resnet20_cifar10_1000samples_1000batches_0seed_allweights.pkl'
dataset = FISHER_PATH.split('_')[1]

#SPARSITY=[0.7,0.8, 0.9, 0.98] # cifar10-resnet20
SPARSITY=[0.8] # cifar10-resnet20
#SPARSITY=[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # cifar10-resnet20
#SPARSITY=[0.9, 0.95]
#RANGE=[1111] # for DEBUG: only one pair each iteration
RANGE = [10,100]
#RANGE=[10, 50, 100, 200]
#MAX_NO_MATCH=[1111] # for DEBUG: only one pair each iteraction
MAX_NO_MATCH=[10,20] # for DEBUG: only one pair each iteraction
THRESHOLD=[1e-3, 1e-4, 1e-5, 1e-6]
#THRESHOLD=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
#THRESHOLD=[5e-6, 1e-5, 5e-5]
GPU=sys.argv[1]
only_cal_delta_w = False
init_method = 'mag'
swap_one_per_iter = False

#greedy_cls = 'greedyblock' #['greedy', 'greedyblock']

if only_cal_delta_w:
    tag_only_cal_delta_w = '--only-delta-w'
else:
    tag_only_cal_delta_w = '--not-only-delta-w'
if swap_one_per_iter:
    tag_swap_one_per_iter = '--swap-one-per-iter'
else:
    tag_swap_one_per_iter = '--not-swap-one-per-iter'

root,_ = os.path.splitext(FISHER_PATH)
root = os.path.join('./logs', root)
if not os.path.exists(root):
    os.makedirs(root)
if only_cal_delta_w:
    f = open(f'run_batch_delta_w_{dataset}_{GPU}.sh', 'w')
else:
    f = open(f'run_batch_{dataset}_{GPU}.sh', 'w')

for s in  SPARSITY:
    for r in RANGE:
        for m in MAX_NO_MATCH:
            for t in THRESHOLD:
                name = get_greedy_exp_name(s, t, r, m)
                log_path = os.path.join(root, name + '.log')
                f.write(f'CUDA_VISIBLE_DEVICES={GPU} python {greedy_cls}.py --sparsity={s} --fisher_path={FISHER_PATH} --range={r} --max-no-match={m} --threshold={t} {tag_only_cal_delta_w} --init_method {init_method} {tag_swap_one_per_iter} >> {log_path} 2>&1 \n')
f.close()

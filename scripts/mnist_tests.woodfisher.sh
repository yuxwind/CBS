NOWDATE=$(date '+%Y_%m_%d_%H:%M:%S') # :%N for further granularity
EXP_NAME=test_code_${NOWDATE}
RESULT_FILE=./tests/results_${NOWDATE}.csv
#greedy_iter=$1
greedy_iter=0
GPU=2
N_arg=$#
if [ $N_arg -eq 0 ]; then
    GPU=2
else
    GPU=$1
fi
echo GPU $GPU

# kfac
CUDA_VISIBLE_DEVICES=$GPU python ./main.py \
    --exp_name=${EXP_NAME} \
    --result-file ${RESULT_FILE} \
    --dset=mnist --dset_path=../datasets \
    --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt \
    --arch=mlpnet \
    --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 \
    --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml \
    --seed 1 --deterministic  \
    `####### set model config` \
    --use-model-config   \
    --not-oldfashioned   \
    `####### when --update-config is set, the following parameters will overwrite the ones in yaml` \
    --update-config  \
    --prune-modules fc1_fc2_fc3 \
    --target-sparsity 0.90 \
    `####### comment out this if using old_w` \
    `#--not-update-weights`  \
    --comb-method 'ep' \
    --greedy-iter $greedy_iter \
    --prune-end 25 --prune-freq 25 \
    --prune-class woodfisher \
    `####### the following parameters are for fisher algorithms` \
    --fisher-damp 1e-3 \
    `####### Get samples used to calculate fisher inverse` \
    --full-subsample  \
    `####### this seems to make no difference to the choice of the data selection...` \
    --fisher-subsample-size 10000 \
    --fisher-mini-bsz 1 \
    `####### ont shot vs. gradual pruning` \
    --one-shot         \
    --woodburry-joint-sparsify

NOWDATE=$(date '+%Y_%m_%d_%H:%M:%S') # :%N for further granularity
EXP_NAME=mlp_mnist_global_magnitude

TARGET_SPARSITY=$1
GPU=$2

SEED=0
ROOT_DIR="."
LOG_DIR="${ROOT_DIR}/prob_regressor_results/${EXP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/prob_regressor_results/${EXP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}
PRUNER=globalmagni
NAME=${PRUNER}-joint_module-all_target_sp-${TARGET_SPARSITY}_oneshot-seed-${SEED}
LOG_NAME=$LOG_DIR$NAME.txt
CSV_NAME=$CSV_DIR$NAME.csv

#greedy_iter=$1
greedy_iter=0

# kfac
CUDA_VISIBLE_DEVICES=$GPU python ./main.py \
    --exp_name=${EXP_NAME} \
    --result-file ${CSV_NAME} \
    --dset=mnist --dset_path=../datasets \
    --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt \
    --arch=mlpnet \
    --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 \
    --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml \
    --seed $SEED --deterministic  \
    `####### set model config` \
    --use-model-config   \
    --not-oldfashioned   \
    `####### when --update-config is set, the following parameters will overwrite the ones in yaml` \
    --update-config  \
    --prune-modules fc1_fc2_fc3 \
    --target-sparsity $TARGET_SPARSITY \
    `####### comment out this if using old_w` \
    `#--not-update-weights`  \
    --comb-method 'ep' \
    --greedy-iter $greedy_iter \
    --prune-end 25 --prune-freq 25 \
    --prune-class ${PRUNER} \
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
#--woodburry-joint-sparsify >  ${LOG_NAME} 2>&1

##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash

NOWDATE=$(date +%F_%H-%M)
NOWDATE='eval_selection${NOWDATE}'
EXP_NAME="greedy_online-cifar10-resnet20_no_allweights"

ROOT_DIR=/uusoc/exports/scratch/xiny/project/networkCompression/WoodFisher
DATA_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/datasets"
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/

MODULE="layer1.0.conv1_layer1.0.conv2_layer1.1.conv1_layer1.1.conv2_layer1.2.conv1_layer1.2.conv2_layer2.0.conv1_layer2.0.conv2_layer2.1.conv1_layer2.1.conv2_layer2.2.conv1_layer2.2.conv2_layer3.0.conv1_layer3.0.conv2_layer3.1.conv1_layer3.1.conv2_layer3.2.conv1_layer3.2.conv2"

ID=0
SEED=0

FISHER_SUBSAMPLE_SIZE=1000
FISHER_MINI_BSZ=1
#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNER="comb"
FISHER_DAMP="1e-5"
EPOCH_END="2"
PROPER_FLAG="1"
ONE_SHOT="--one-shot"
#CKP_PATH="${ROOT_DIR}/checkpoints/resnet20_cifar10.pth.tar"
CKP_PATH="${ROOT_DIR}/../exp_root/exp_oct_21_resnet20_all_rep/20211026_18-55-28_694095_239/regular_checkpoint.ckpt"

SPARSITY=$1
THRESHOLD=$2
RANGE=$3
MAX_NO_MATCH=$4
GREEDY_PATH=$5
GPU=$6
args_update_weights="--not-update-weights"
SWAP_FLAG="--not-swap-one-per-iter"
WHEN_TO_GREEDY="offline"
N_FLUCTATION=5


ARCH_NAME=resnet20_cifar10_1000samples_1000batches_0seed
WGH_PATH=$WEIGHT_DIR/resnet20_cifar10_1000samples_1000batches_0seed.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/resnet20_cifar10_1000samples_1000batches_0seed-idx_2_module.npy

name="sparsity${SPARSITY}_T${THRESHOLD}_Range${RANGE}_NOMATCH${MAX_NO_MATCH}"

WOODFISHER_MASK_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv/global_mask_sparsity${SPARSITY}.pkl

LOG_DIR="${EXP_DIR}/${EXP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${EXP_DIR}/${EXP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
echo $LOG_DIR
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

GREEDY_DIR=$EXP_DIR/$EXP_NAME/results.${NOWDATE}/$ARCH_NAME
LOG_PATH=$LOG_DIR/${name}.log
CSV_PATH=$CSV_DIR/${name}.csv

GREEDY_METHOD=greedyblock
GREEDY_INIT=woodfisherblock #[mag, wg, woodfisherblock]
MAX_ITER=100


greedy_args=" 
$args_update_weights \
--greedy-dir ${GREEDY_DIR} \
--idx-2-module-path ${IDX_2_MODULE_PATH} \
--wgh_path ${WGH_PATH} \
--greedy_method ${GREEDY_METHOD} \
--init_method ${GREEDY_INIT} \
--target-sparsity ${SPARSITY} \
--max-iter ${MAX_ITER} \
--range ${RANGE} \
--max-no-match ${MAX_NO_MATCH} \
--threshold ${THRESHOLD} \
${SWAP_FLAG} \
--max-N-fluctation ${N_FLUCTATION} \
--woodfisher_mask_path ${WOODFISHER_MASK_PATH} \
--when-to-greedy ${WHEN_TO_GREEDY} \
--greedy-path ${GREEDY_PATH} \
"

args="
$ONE_SHOT \
--exp_name=${EXP_NAME}  \
--dset=cifar10 \
--dset_path=${DATA_DIR} \
--arch=resnet20 \
--config_path=${ROOT_DIR}/configs/resnet20_woodfisher.yaml \
--workers=1 \
--batch_size=64 \
--logging_level debug \
--gpus=0 \
--from_checkpoint_path=${CKP_PATH} \
--batched-test \
--not-oldfashioned \
--disable-log-soft \
--use-model-config \
--sweep-id ${ID} \
--fisher-damp ${FISHER_DAMP} \
--prune-modules ${MODULE} \
--fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} \
--fisher-mini-bsz ${FISHER_MINI_BSZ} \
--update-config \
--prune-class ${PRUNER} \
--prune-end ${EPOCH_END} \
--prune-freq ${EPOCH_END} \
--result-file ${CSV_PATH} \
--seed ${SEED} \
--deterministic \
--full-subsample \
--fisher-optimized \
--fisher-parts 5 \
--offload-grads \
--offload-inv \
"

REDIRECT_LOG=$6
if [ "${REDIRECT_LOG}" = "redirect_log" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT_DIR}/main.py $args $greedy_args 
fi

##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash

script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME

# if NOWDATE is passed by args, use it; otherwise tag this exp as a test 
if [ "$#" -eq 7 ] && [ $7 != test ]; then
    NOWDATE=$7
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

echo $NOWDATE

ROOT_DIR=/uusoc/exports/scratch/xiny/project/networkCompression/WoodFisher
DATA_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/datasets"
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/

# create snapshot for code and scripts if , and run the codes from the snapshot
SNAPSHOT_DIR=$ROOT_DIR/code-snapshot/$EXP_NAME.${NOWDATE}
if [ ! -d $SNAPSHOT_DIR ]; then
    sh $ROOT_DIR/scripts/code-snapshot.sh $script_name $NOWDATE
fi
CODE_DIR=$SNAPSHOT_DIR


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
SCALE_PRUNE_UPDATE=$5
GPU=$6
#args_update_weights="--not-update-weights"
args_update_weights=""
SWAP_FLAG="--not-swap-one-per-iter"
N_FLUCTATION=5
WHEN_TO_GREEDY="online"
KEEP_PRUNED_FLAG="--keep-pruned"

#add more info for exp
NOWDATE=${NOWDATE}.scale_update_${SCALE_PRUNE_UPDATE}

ARCH_NAME=resnet20_cifar10_1000samples_1000batches_0seed
WGH_PATH=$WEIGHT_DIR/resnet20_cifar10_1000samples_1000batches_0seed.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/resnet20_cifar10_1000samples_1000batches_0seed-idx_2_module.npy
FISHER_INV_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv

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
GREEDY_INIT=random #[mag, wg, woodfisherblock, random]
MAX_ITER=100
WEIGHT_UPDATE_METHOD="multiple"
UPDATE_WEIGHT_ONLINE="--update-weight-online"
ABLATION_STUDY_UPDATE_WEIGHT="--ablation-study-update-weight"


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
${KEEP_PRUNED_FLAG}  \
--weight_update_method ${WEIGHT_UPDATE_METHOD} \
${UPDATE_WEIGHT_ONLINE} \
${ABLATION_STUDY_UPDATE_WEIGHT} \
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
--fisher_inv_path ${FISHER_INV_PATH} \
--scale-prune-update ${SCALE_PRUNE_UPDATE} \
"

if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args 
fi

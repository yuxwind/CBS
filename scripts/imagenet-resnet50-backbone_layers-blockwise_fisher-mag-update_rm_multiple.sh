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

ROOT_DIR="."
DATA_DIR="../datasets/ILSVRC"
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data
EXP_DIR=$ROOT_DIR/prob_regressor_results
CONFIG_PATH=./configs/resnet50.yaml

# create snapshot for code and scripts if , and run the codes from the snapshot
SNAPSHOT_DIR=$ROOT_DIR/code-snapshot/$EXP_NAME.${NOWDATE}
if [ ! -d $SNAPSHOT_DIR ]; then
    sh $ROOT_DIR/scripts/code-snapshot.sh $script_name $NOWDATE
fi
CODE_DIR=$SNAPSHOT_DIR


ID=0
SEED=0

DATASET=imagenet
MODEL=resnet50
FITTABLE=1000
FISHER_SUBSAMPLE_SIZE=240
FISHER_MINI_BSZ=100
#BSZ=256
BSZ=64
EPOCHS=100

#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNER="comb"
FISHER_DAMP="1e-5"
EPOCH_END="2"
ONE_SHOT="--one-shot"
CKP_PATH="./checkpoints/ResNet50-STR-Dense.pth"

SPARSITY=$1
THRESHOLD=$2
RANGE=$3
MAX_NO_MATCH=$4
SCALE_PRUNE_UPDATE=$5
GPU=$6

# for combinatorial pruning
#args_update_weights="--not-update-weights"
args_update_weights=""
SWAP_FLAG="--not-swap-one-per-iter"
N_FLUCTATION=5
WHEN_TO_GREEDY="online"
KEEP_PRUNED_FLAG="--keep-pruned"
# for greedy pruning 
GREEDY_METHOD=greedyblock
GREEDY_INIT=mag #[mag, wg, woodfisherblock]
MAX_ITER=100
WEIGHT_UPDATE_METHOD="multiple"
UPDATE_WEIGHT_ONLINE="--update-weight-online"
ABLATION_STUDY_UPDATE_WEIGHT="--ablation-study-update-weight"

name="sparsity${SPARSITY}_T${THRESHOLD}_Range${RANGE}_NOMATCH${MAX_NO_MATCH}"


#add more info for exp
NOWDATE=${NOWDATE}.scale_update_${SCALE_PRUNE_UPDATE}

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=resnet50_imagenet_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
WGH_PATH=$WEIGHT_DIR/${ARCH_NAME}.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/${ARCH_NAME}-idx_2_module.npy
FISHER_INV_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv


WOODFISHER_MASK_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv/global_mask_sparsity${SPARSITY}.pkl

LOG_DIR="${EXP_DIR}/${EXP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${EXP_DIR}/${EXP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
echo $LOG_DIR
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

GREEDY_DIR=$EXP_DIR/$EXP_NAME/results.${NOWDATE}/$ARCH_NAME
LOG_PATH=$LOG_DIR/${name}.log
CSV_PATH=$CSV_DIR/${name}.csv


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
--fisher_inv_path ${FISHER_INV_PATH} \
--scale-prune-update ${SCALE_PRUNE_UPDATE} \
"

args="
$ONE_SHOT \
--exp_name=${EXP_NAME}  \
--dset=${DATASET} \
--dset_path=${DATA_DIR} \
--arch=${MODEL} \
--config_path=${CONFIG_PATH} \
--workers=20 \
--batch_size=${BSZ} \
--logging_level info \
--pretrained  \
--from_checkpoint_path=${CKP_PATH} \
--batched-test \
--not-oldfashioned \
--disable-log-soft \
--use-model-config \
--sweep-id ${ID} \
--fisher-damp ${FISHER_DAMP} \
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
--fisher-split-grads \
--fittable-params $FITTABLE \
--woodburry-joint-sparsify \
--eval-fast \
--epochs $EPOCHS \
--gpus 0 \
"

if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
    #python ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args 
    #python ${CODE_DIR}/main.py $args $greedy_args 
fi

##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash
#NOWDATE=$(date '+%Y_%m_%d') # :%N for further granularity
script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME

SPARSITY=$1
SEED=$2
GPU=$3
FLAG_GPUS="--gpus=0"
#FLAG_GPUS="--cpu"

DATASET=cifar10
MODEL=resnet20

if [ "$#" -eq 4 ] && [ $4 != test ]; then
    NOWDATE=$4
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

ARCH_NAME=resnet20_cifar10_${SEED}seed
name="sparsity${SPARSITY}"

ROOT_DIR=./
DATA_DIR=${ROOT_DIR}/../datasets
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/
CODE_DIR='./'
SWEEP_NAME=$EXP_NAME
LOG_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

RESULT_PATH="${CSV_DIR}/${name}.csv"
LOG_PATH="${LOG_DIR}/${name}.log"

MODULE="layer1.0.conv1_layer1.0.conv2_layer1.1.conv1_layer1.1.conv2_layer1.2.conv1_layer1.2.conv2_layer2.0.conv1_layer2.0.conv2_layer2.1.conv1_layer2.1.conv2_layer2.2.conv1_layer2.2.conv2_layer3.0.conv1_layer3.0.conv2_layer3.1.conv1_layer3.1.conv2_layer3.2.conv1_layer3.2.conv2"

FISHER_SUBSAMPLE_SIZE=1000
#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNER=woodfisherblock
JOINT=1
FISHER_DAMP="1e-5"
EPOCH_END="2"
PROPER_FLAG="1"
DQT='"'
JOINT_FLAG="--woodburry-joint-sparsify"
CKP_PATH="${ROOT_DIR}/checkpoints/resnet20_cifar10.20211026_18-55-28_694095_239.regular_checkpoint.ckpt"
ONE_SHOT="--one-shot"

extra_cmd=" $ONE_SHOT "

ID=0

args="--exp_name=${SWEEP_NAME}  \
    --dset=cifar10 \
    --dset_path=${DATA_DIR} \
    --arch=resnet20 \
    --config_path=${ROOT_DIR}/configs/resnet20_woodfisher.yaml \
    --workers=1 \
    --batch_size=64 \
    --logging_level debug \
    ${FLAG_GPUS} \
    --from_checkpoint_path=${CKP_PATH} \
    --batched-test \
    --not-oldfashioned \
    --disable-log-soft \
    --use-model-config \
    --fisher-damp ${FISHER_DAMP} \
    --prune-end ${EPOCH_END} --prune-freq ${EPOCH_END} \
    --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv \
    --fisher-optimized --fisher-parts 5 --offload-grads --offload-inv \
    $extra_cmd \
"
FISHER_MINIBSZ=1
cfg_args="
    --update-config --prune-class ${PRUNER} \
    --deterministic --full-subsample ${JOINT_FLAG} \
    --seed ${SEED} \
    --prune-modules ${MODULE} \
    --target-sparsity ${SPARSITY} \
    --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} \
    --fisher-mini-bsz ${FISHER_MINIBSZ} \
    --sweep-id ${ID} \
    --not-update-weights \
"
                            
if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU}  python ${ROOT_DIR}/main.py $args $cfg_args > ${LOG_PATH} 2>&1 
else
    CUDA_VISIBLE_DEVICES=${GPU}  python ${ROOT_DIR}/main.py $args $cfg_args 
fi

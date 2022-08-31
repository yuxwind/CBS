##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash

script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME

TARGET_SPARSITY=$1
GPU=$2
# if NOWDATE is passed by args, use it; otherwise tag this exp as a test
if [ "$#" -eq 3 ] && [ $3 != test ]; then
    NOWDATE=$3
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

ROOT_DIR="."
DATA_DIR="$ROOT_DIR/../datasets"
#ROOT_DIR=/uusoc/exports/scratch/xiny/project/networkCompression/WoodFisher
#DATA_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/datasets"
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/

# create snapshot for code and scripts if , and run the codes from the snapshot
SNAPSHOT_DIR=$ROOT_DIR/code-snapshot/$EXP_NAME.${NOWDATE}
if [ ! -d $SNAPSHOT_DIR ]; then
    sh $ROOT_DIR/scripts/code-snapshot.sh $script_name $NOWDATE
fi
CODE_DIR=$SNAPSHOT_DIR

#TARGET_SPARSITYS=(0.2 0.4 0.6 0.7 0.8 0.9 0.1 0.3 0.5 0.98)
#TARGET_SPARSITYS=(0.75 0.857 0.875 0.933 0.95 0.967 0.98 0.986 0.99)
MODULES=("conv1_conv2_conv3_fc1_fc2")

#SEEDS=(0 1 2 3 4)
SEEDS=(0)
FISHER_SUBSAMPLE_SIZES=(4000)
#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNERS=(globalmagni)
JOINTS=(1)
FISHER_DAMP="1e-5"
EPOCH_END="2"
PROPER_FLAG="1"
SWEEP_NAME=${EXP_NAME}
DQT='"'
LOG_DIR="${ROOT_DIR}/prob_regressor_results/${EXP_NAME}/log.${NOWDATE}"
CSV_DIR="${ROOT_DIR}/prob_regressor_results/${EXP_NAME}/csv.${NOWDATE}"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}
CKP_PATH="${ROOT_DIR}/checkpoints/cifarnet_from_scratch_20220106_14-20-35_486324_0-best_checkpoint.ckpt"
ONE_SHOT="--one-shot"
extra_cmd=" $ONE_SHOT"

ID=0


for PRUNER in "${PRUNERS[@]}"
do
    for JOINT in "${JOINTS[@]}"
    do
        if [ "${JOINT}" = "0" ]; then
            JOINT_FLAG=""
        elif [ "${JOINT}" = "1" ]; then
            JOINT_FLAG="--woodburry-joint-sparsify"
        fi

        for SEED in "${SEEDS[@]}"
        do
            for MODULE in "${MODULES[@]}"
            do
                    for FISHER_SUBSAMPLE_SIZE in "${FISHER_SUBSAMPLE_SIZES[@]}"
                    do
                        if [ "${FISHER_SUBSAMPLE_SIZE}" = 80 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 1000 ]; then
                            #FISHER_MINIBSZS=(1 50)
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 4000 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 5000 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 8000 ]; then
                            FISHER_MINIBSZS=(10)
                        fi

                        for FISHER_MINIBSZ in "${FISHER_MINIBSZS[@]}"
                        do

args="--exp_name=${SWEEP_NAME} \
    --dset=cifar10 \
    --dset_path=${DATA_DIR} \
    --arch=cifarnet\
    --config_path=${ROOT_DIR}/configs/cifarnet_retrain.yaml \
    --workers=1 \
    --batch_size=64 \
    --logging_level debug \
    --gpus=0 \
    --batched-test \
    --not-oldfashioned \
    --disable-log-soft \
    --use-model-config \
    --sweep-id ${ID} \
    --fisher-damp ${FISHER_DAMP} \
    --prune-modules ${MODULE}\
    --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} \
    --fisher-mini-bsz ${FISHER_MINIBSZ} \
    --update-config \
    --prune-class ${PRUNER} \
    --target-sparsity ${TARGET_SPARSITY} \
    --prune-end ${EPOCH_END} \
    --prune-freq ${EPOCH_END} \
    --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv 
    --seed ${SEED} \
    --deterministic --full-subsample ${JOINT_FLAG} \
    --fisher-optimized \
    --fisher-parts 5   \
    --offload-grads \
    --offload-inv \
    --from_checkpoint_path=${CKP_PATH} \
    $extra_cmd  \
" 

log=${LOG_DIR}/sparsity${TARGET_SPARSITY}.log
echo $log
if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT_DIR}/main.py $args &> ${log} 2>&1 
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT_DIR}/main.py $args 
fi

                    done
                done
            done
        done
    done
done

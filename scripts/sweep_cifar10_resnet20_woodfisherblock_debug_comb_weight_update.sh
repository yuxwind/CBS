##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash

script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"

#TARGET_SPARSITYS=(0.6 0.7 0.8 0.9 0.98)
#TARGET_SPARSITYS=(0.1 0.2 0.3 0.4 0.5)
TARGET_SPARSITYS=(0.8)
#TARGET_SPARSITYS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.98)
#TARGET_SPARSITYS=(0.9 0.98)
MODULES=("layer1.0.conv1_layer1.0.conv2_layer1.1.conv1_layer1.1.conv2_layer1.2.conv1_layer1.2.conv2_layer2.0.conv1_layer2.0.conv2_layer2.1.conv1_layer2.1.conv2_layer2.2.conv1_layer2.2.conv2_layer3.0.conv1_layer3.0.conv2_layer3.1.conv1_layer3.1.conv2_layer3.2.conv1_layer3.2.conv2")

GPUS=($1)
#SEEDS=(0 1 2)
SEEDS=(0)
FISHER_SUBSAMPLE_SIZES=(1000)
#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNERS=(woodfisherblock)
JOINTS=(1)
FISHER_DAMP="1e-5"
EPOCH_END="2"
PROPER_FLAG="1"
ROOT_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/WoodFisher"
DATA_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/datasets"
#SWEEP_NAME="exp_resnet20_backbone_woodfisherblock_no_weight_update"${NOWDATE}
SWEEP_NAME=$EXP_NAME.$NOWDATE
DQT='"'
LOG_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}
ONE_SHOT="--one-shot"
#CKP_PATH="${ROOT_DIR}/checkpoints/resnet20_cifar10.pth.tar"
CKP_PATH="${ROOT_DIR}/../exp_root/exp_oct_21_resnet20_all_rep/20211026_18-55-28_694095_239/regular_checkpoint.ckpt"

#pruning_path="prob_regressor_results/greedy/resnet20_cifar10_1000samples_1000batches_0seed/sparsity0.60_T1.00e-04_Range100_NOMATCH20-iter_48.npy"
extra_cmd=" $ONE_SHOT "

ID=0

# Extra params for GreedyBlockPruner 
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/

ARCH_NAME=resnet20_cifar10_1000samples_1000batches_0seed
IDX_2_MODULE_PATH=$WEIGHT_DIR/$ARCH_NAME-idx_2_module.npy
WGH_PATH=$WEIGHT_DIR/$ARCH_NAME.pkl
FISHER_INV_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv

GREEDY_METHOD=greedyblock
INIT_METHOD=mag  #mag, wg, woodfisherblock
MAX_ITER=100
THRESHOLD=1
RANGE=1
MAX_NO_MATCH=1
GREEDY_PATH=$EXP_DIR/cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers/results.train_loss_all_samples.20211202.22_52/resnet20_cifar10_1000samples_1000batches_0seed/sparsity0.80_T1.00e-05_Range10_NOMATCH10-magnitude.npy
SPARSITY=0.8
FLAG_ONLY_DELTA_W='--only-delta-w'
FLAG_SWAP_ONE_PER_ITER='--not-swap-one-per-iter'

WOODFISHER_MASK_PATH=$WEIGHT_DIR/$ARCH_NAME.fisher_inv/global_mask_sparsity${SPARSITY}.pkl
GREEDY_DIR="$EXP_DIR/$EXP_NAME/results.${NOWDATE}/$ARCH_NAME"
mkdir -p ${LOG_DIR}
mkdir -p ${GREEDY_DIR}

greedy_args="
    --greedy-dir=${GREEDY_DIR} \
    --idx-2-module-path=${IDX_2_MODULE_PATH} \
    --wgh_path=${WGH_PATH} \
    --fisher_inv_path=${FISHER_INV_PATH} \
    --range=${RANGE} \
    --max-no-match=${MAX_NO_MATCH} \
    --threshold=${THRESHOLD} \
    --init_method ${INIT_METHOD} \
    ${FLAG_SWAP_ONE_PER_ITER} \
    --greedy-path=${GREEDY_PATH} \
"

extra_cmd=" $extra_cmd $greedy_args "

args="--exp_name=${SWEEP_NAME}  \
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
    --fisher-damp ${FISHER_DAMP} \
    --prune-end ${EPOCH_END} --prune-freq ${EPOCH_END} \
    --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv \
    --fisher-optimized --fisher-parts 5 --offload-grads --offload-inv \
    $extra_cmd \
"

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
                for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
                do
                    for FISHER_SUBSAMPLE_SIZE in "${FISHER_SUBSAMPLE_SIZES[@]}"
                    do
                        if [ "${FISHER_SUBSAMPLE_SIZE}" = 80 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 1000 ]; then
                            #FISHER_MINIBSZS=(1 50)
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 5000 ]; then
                            FISHER_MINIBSZS=(1)
                        fi

                        for FISHER_MINIBSZ in "${FISHER_MINIBSZS[@]}"
                        do
                            cfg_args="
                                --update-config --prune-class ${PRUNER} \
                                --deterministic --full-subsample ${JOINT_FLAG} \
                                --seed ${SEED} \
                                --prune-modules ${MODULE} \
                                --target-sparsity ${TARGET_SPARSITY} \
                                --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} \
                                --fisher-mini-bsz ${FISHER_MINIBSZ} \
                                --sweep-id ${ID} \
                            "
LOG_PATH=${LOG_DIR}/${PRUNER}_proper-1_joint-${JOINT}_module-all_target_sp-${TARGET_SPARSITY}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_damp-${FISHER_DAMP}_seed-${SEED}.txt
                            
CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py $args $cfg_args 
#CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py $args $cfg_args > ${LOG_PATH} 2>&1 
                            
                            ID=$((ID+1))

                        done
                    done
                done
            done
        done
    done
done

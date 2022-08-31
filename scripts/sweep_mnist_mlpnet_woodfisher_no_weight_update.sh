
script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME
ARCH=mlpnet
DATASET=mnist

SPARSITY=$1
SEED=$2
GPU=$3

if [ "$#" -eq 4 ] && [ $4 != test ]; then
    NOWDATE=$4
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

#SEED=0

ROOT_DIR=./
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/
CODE_DIR='./'

DATASET=mnist
MODEL=mlpnet
DATA_PATH=../datasets/
CONFIG_PATH=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml
PRUNER=woodfisher
EPOCHS=100
FISHER_SUBSAMPLE_SIZE=10000
FISHER_MINI_BSZ=1
LOAD_FROM="./checkpoints/mnist_25_epoch_93.97.ckpt"
BSZ=64

MODULE=fc1_fc2_fc3
N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${ARCH}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
WGH_PATH=$WEIGHT_DIR/${ARCH_NAME}.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/${ARCH_NAME}-idx_2_module.npy


name="sparsity${SPARSITY}"

LOG_DIR="${EXP_DIR}/${EXP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${EXP_DIR}/${EXP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
mkdir -p ${CSV_DIR}
mkdir -p ${LOG_DIR}
RESULT_PATH="${CSV_DIR}/${name}.csv"
LOG_PATH="${LOG_DIR}/${name}.log"

ONE_SHOT="--one-shot"
SCALE_PRUNE_UPDATE=0.9

echo "EXPERIMENT $EXP_NAME"
export PYTHONUNBUFFERED=1

args="
--prune-modules ${MODULE} \
--exp_name=$EXP_NAME \
--dset=$DATASET \
--dset_path=$DATA_PATH \
--arch=$MODEL \
--config_path=$CONFIG_PATH \
--workers=1 --batch_size=${BSZ} --logging_level info \
--from_checkpoint_path $LOAD_FROM \
--batched-test --not-oldfashioned --use-model-config \
--sweep-id 20 --fisher-damp 1e-3 --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINI_BSZ} --update-config --prune-class $PRUNER \
--target-sparsity $SPARSITY \
--seed ${SEED} --full-subsample \
--woodburry-joint-sparsify \
${ONE_SHOT} \
--result-file $RESULT_PATH --epochs $EPOCHS \
--scale-prune-update ${SCALE_PRUNE_UPDATE} \
--not-update-weights \
"
if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args
fi


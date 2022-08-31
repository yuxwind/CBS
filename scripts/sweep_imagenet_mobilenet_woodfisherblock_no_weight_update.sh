
script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME

TARGET=$1
SEED=$2
GPU=$3


if [ "$#" -eq 4 ] && [ $4 != test ]; then
    NOWDATE=$4
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

ROOT_DIR=./
DATA_PATH=${ROOT_DIR}/../datasets/ILSVRC
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/
CODE_DIR='./'

DATASET=imagenet
MODEL=mobilenet
CONFIG_PATH=./configs/mobilenetv1.yaml
#PRUNER=woodfisherblockdynamic
PRUNER=woodfisherblock
FITTABLE=10000
EPOCHS=100
FISHER_SUBSAMPLE_SIZE=400
FISHER_MINI_BSZ=2400
MAX_MINI_BSZ=800
LOAD_FROM="./checkpoints/MobileNetV1-Dense-STR.pth"
BSZ=256

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${MODEL}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
#ARCH_NAME="seed${SEED}_batchsize${BSZ}_fittable${FITTABLE}"
name="sparsity${TARGET}"

LOG_DIR="${EXP_DIR}/${EXP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${EXP_DIR}/${EXP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
mkdir -p ${CSV_DIR}
mkdir -p ${LOG_DIR}
RESULT_PATH="${CSV_DIR}/${name}.csv"
LOG_PATH="${LOG_DIR}/${name}.log"

ONE_SHOT="--one-shot"
SCALE_PRUNE_UPDATE=0.9
args_update_weights="--not-update-weights"
#args_update_weights=""

echo "EXPERIMENT $EXP_NAME"
export PYTHONUNBUFFERED=1

args="
--exp_name=$EXP_NAME \
--dset=$DATASET \
--dset_path=$DATA_PATH \
--arch=$MODEL \
--config_path=$CONFIG_PATH \
--workers=24 --batch_size=${BSZ} --logging_level info \
--pretrained --from_checkpoint_path $LOAD_FROM \
--batched-test --not-oldfashioned --disable-log-soft --use-model-config \
--sweep-id 20 --fisher-damp 1e-5 --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINI_BSZ} --update-config --prune-class $PRUNER \
--target-sparsity $TARGET \
--seed ${SEED} --full-subsample --fisher-split-grads --fittable-params $FITTABLE \
--woodburry-joint-sparsify --offload-inv --offload-grads \
${ONE_SHOT} \
--result-file $RESULT_PATH --epochs $EPOCHS --eval-fast \
--scale-prune-update ${SCALE_PRUNE_UPDATE} \
--max-mini-bsz ${MAX_MINI_BSZ} \
${args_update_weights}  \
"
if [ "$is_test" -eq 0 ] ; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
else
    CUDA_VISIBLE_DEVICES=${GPU} python ${CODE_DIR}/main.py $args $greedy_args
fi


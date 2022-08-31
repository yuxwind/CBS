#ROOT_DIR=/uusoc/exports/scratch/xiny/project/networkCompression/WoodFisher
#DATA_DIR="/uusoc/exports/scratch/xiny/project/networkCompression/datasets"
ROOT_DIR="."
DATA_DIR="$ROOT_DIR/../datasets"

#cifarnet
ARCH=cifarnet
DATASET=cifar10
MODULE="conv1_conv2_conv3_fc1_fc2"

CKP_PATH="${ROOT_DIR}/checkpoints/cifarnet_from_scratch_20220106_14-20-35_486324_0-best_checkpoint.ckpt"
CONFIG_PATH=${ROOT_DIR}/configs/cifarnet_retrain.yaml

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${ARCH}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed

WGH_PATH=$WEIGHT_DIR/${ARCH_NAME}.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/${ARCH_NAME}-idx_2_module.npy
FISHER_INV_PATH=$WEIGHT_DIR/${ARCH_NAME}.fisher_inv

--dset=${DATASET} \
--dset_path=${DATA_DIR} \
--arch=${ARCH} \
--config_path=${CONFIG_PATH} \


# imagenet
DATA_DIR="${ROOT_DIR}/../datasets/ILSVRC"
CONFIG_PATH=./configs/resnet50.yaml

DATASET=imagenet
MODEL=resnet50
FITTABLE=1000
FISHER_SUBSAMPLE_SIZE=240
FISHER_MINI_BSZ=100
#BSZ=256
BSZ=64
EPOCHS=100

CKP_PATH="${ROOT_DIR}/./checkpoints/ResNet50-STR-Dense.pth"

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${MODEL}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
WGH_PATH=$WEIGHT_DIR/${ARCH_NAME}.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/${ARCH_NAME}-idx_2_module.npy


--workers=20 \
--pretrained  \ # return a pretrained model 
--prune-modules ${MODULE} \ $ Delete this line because it is set in $CONFIG_PATH
--fittable-params ${FITTABLE} \
--woodburry-joint-sparsify \
--epochs $EPOCHS \
--eval-fast \    # use dataloader()
--batched-test \ # test in batch mode. Used in my_test_testset() which doesn't use dataloader()
--fisher-split-grads \ # what is this?

# mobileNet

DATASET=imagenet
MODEL=mobilenet
DATA_PATH=../datasets/ILSVRC
CONFIG_PATH=./configs/mobilenetv1.yaml
FITTABLE=10000
EPOCHS=100
FISHER_SUBSAMPLE=400
FISHER_MINI_BSZ=2400
MAX_MINI_BSZ=800
LOAD_FROM="./checkpoints/MobileNetV1-Dense-STR.pth"
--workers=24
--dset=$DATASET \
--arch=$MODEL \
--dset_path=$DATA_PATH \
--config_path=$CONFIG_PATH \
--batch_size=128
--batched-test
--disable-log-soft
--max-mini-bsz ${MAX_MINI_BSZ} \
--fittable-params $FITTABLE \
 --offload-inv --offload-grads \
 --fisher-damp 1e-5


# mlpnet
DATASET=mnist
MODEL=mlpnet

CKP_PATH="${ROOT_DIR}/checkpoints/mnist_25_epoch_93.97.ckpt"
MODULE="fc1_fc2_fc3"
FISHER_DAMP="1e-3"
FISHER_SUBSAMPLE_SIZE=10000
FISHER_MINI_BSZ=1
GREEDY_METHOD=greedy

NOTE: remove 
--disable-log-soft
--fisher-split-grads
--offload-inv 
--offload-grads
 --eval-fast
 --max-mini-bsz ${MAX_MINI_BSZ}

Change those:
--dset=mnist \
--arch=mlpnet\
--workers=1
--batch_size=64
--config_path=${ROOT_DIR}/configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml \

# ResNet20
DATASET=cifar10
MODEL=resnet20

FISHER_SUBSAMPLE_SIZE=1000
FISHER_MINI_BSZ=1
#BSZ=256
BSZ=64
EPOCHS=100
FISHER_DAMP="1e-5"
CKP_PATH="${ROOT_DIR}/resnet20_cifar10.20211026_18-55-28_694095_239.regular_checkpoint.ckpt"
## NOTE: the ckeckpoint below can't be used as it is a sparse model with sparisty of 60%. WHY? 
#CKP_PATH="${ROOT_DIR}/checkpoints/resnet20_cifar10.train_from_scratch.20211215_16-45-35_229250_0.best_checkpoint.ckpt"

CONFIG_PATH=./configs/resnet20_woodfisher.yaml
MODULE="layer1.0.conv1_layer1.0.conv2_layer1.1.conv1_layer1.1.conv2_layer1.2.conv1_layer1.2.conv2_layer2.0.conv1_layer2.0.conv2_layer2.1.conv1_layer2.1.conv2_layer2.2.conv1_layer2.2.conv2_layer3.0.conv1_layer3.0.conv2_layer3.1.conv1_layer3.1.conv2_layer3.2.conv1_layer3.2.conv2"

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${MODEL}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
WGH_PATH=$WEIGHT_DIR/${ARCH_NAME}.pkl
IDX_2_MODULE_PATH=$WEIGHT_DIR/${ARCH_NAME}-idx_2_module.npy

--workers=1 \
--fisher-parts 5 \
--offload-grads --offload-inv \
--fisher-mini-bsz ${FISHER_MINI_BSZ} \
--fisher-optimized \

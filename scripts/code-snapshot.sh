script_name=$1
NOWDATE=$2

EXP_NAME="${script_name%.*}"
ROOT_DIR='.'
SNAPSHOT_DIR=$ROOT_DIR/code-snapshot/$EXP_NAME.${NOWDATE}

mkdir -p $SNAPSHOT_DIR
mkdir -p $SNAPSHOT_DIR/scripts
mkdir -p $SNAPSHOT_DIR/greedy_alg

cp $ROOT_DIR/scripts/*.sh   $SNAPSHOT_DIR/scripts
cp $ROOT_DIR/scripts/*.py   $SNAPSHOT_DIR/scripts/
cp $ROOT_DIR/env.py         $SNAPSHOT_DIR/
cp $ROOT_DIR/env_cfg.py         $SNAPSHOT_DIR/
cp $ROOT_DIR/main.py        $SNAPSHOT_DIR/
cp $ROOT_DIR/options.py     $SNAPSHOT_DIR/
cp $ROOT_DIR/greedy_alg/*.py $SNAPSHOT_DIR/greedy_alg/
cp $ROOT_DIR/greedy_alg/scripts/*.py $SNAPSHOT_DIR/greedy_alg/scripts
cp $ROOT_DIR/greedy_alg/scripts/*.sh $SNAPSHOT_DIR/greedy_alg/scripts
rsync -avr --exclude=$ROOT_DIR/pruner/__pycache__ $ROOT_DIR/pruners     $SNAPSHOT_DIR/
rsync -avr --exclude=$ROOT_DIR/policies/__pycache__ $ROOT_DIR/policies    $SNAPSHOT_DIR/
rsync -avr --exclude=$ROOT_DIR/common/__pycache__ $ROOT_DIR/common    $SNAPSHOT_DIR/
rsync -avr --exclude=$ROOT_DIR/models/__pycache__ $ROOT_DIR/models    $SNAPSHOT_DIR/
rsync -avr --exclude=$ROOT_DIR/utils/__pycache__ -r $ROOT_DIR/utils     $SNAPSHOT_DIR/

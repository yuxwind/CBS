# Run this script before using the repo so that imports work as expected
export PYTHONPATH=$PYTHONPATH:$PWD
pip install tb-nightly
pip install future
#conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
pip install tqdm
pip install pyyaml
pip install sklearn
pip install matplotlib

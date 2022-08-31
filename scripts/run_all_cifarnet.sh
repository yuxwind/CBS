# baseline1: mag
python scripts/run_batch_baseline.py scripts/cifar10-cifarnet-backbone_layers-mag.sh test
# baseline2: woodfisherblock 
python scripts/run_batch_baseline.py scripts/sweep_cifar10_cifarnet_woodfisherblock.sh test
# baseline2.1: woodfisher without weight update
python scripts/run_batch_baseline.py scripts/sweep_cifar10_cifarnet_woodfisherblock_no_weight_update.sh test
## After baseline2.1, you need to put the global_mask files to prob_regressord_data/$ARCH_NAME.fisher_inv 
## TODO: dump the global_mask to it in the python code...


# greedy1: greedy init=mag
python scripts/run_batch_greedy.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers.sh test
# greedy2: greedy init=woodfisehr
python scripts/run_batch_greedy.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_woodfisherblock_all_layers.sh 1 test


# weight update: run  scale_update=1.0 firstly; then repeat the following scripts with scale_update=0.0 to 0.9

# ablation_study1.2: mag + multiple
#   set thresholds,ranges, max_no_match as place-holder in scripts/run_batch_scale_update.py 
python scripts/run_batch_scale_update.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-mag-update_rm_multiple.sh 0 test
# ablation_study1.1: mag + single
python scripts/run_batch_scale_update.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-mag-update_rm_single.sh 0 test

# ablation_study2.2: mag + multiple
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-woodfisher-update_rm_multiple.sh 1 test
# ablation_study2.1: mag + single
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-woodfisher-update_rm_single.sh 1 test

# weight update for greedy: this requires to set greedy_path as greedy1 in scripts/run_batch_update_weights_with_scale.py
# weightupdaet3.2: greedy init=mag + multiple
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_mag-update_rm_multiple.sh 0 test
# weightupdaet3.1: greedy init=mag + single
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_mag-update_rm_multiple.sh 0 test

# weight update for greedy: this requires to set greedy_path as greedy2 in scripts/run_batch_update_weights_with_scale.py
# weightupdaet3.2: greedy woodfisher=mag + multiple
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_woodfisher-update_rm_multiple.sh 1 test
# weightupdaet3.1: greedy init=mag + single
python scripts/run_batch_update_weights_with_scale.py scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_woodfisher-update_rm_single.sh 1 test

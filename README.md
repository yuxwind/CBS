# CBS
Official Code of The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks[ICML2022]

1. Setup the enviroment:
    ```
    conda create -n cbs python=3.6.13
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    sh setup.sh
    ```
2. Prepare data
    ```
    mkdir prob_regressor_data
    mkdir prob_regressor_results
    mkdir checkpoints
    ```
    Please download checkpoints from [here](https://drive.google.com/drive/folders/18ix239cy261ug_IGZbhtYKPzkkniTyee?usp=sharing). 

3. Run CBS on MLPNet:

    3.1 Run on woodfisher firstly. This will sample data and dump their gradients and Hessien inverse matrix to './prob_regressor_data' for future usage.
        
    ```
        sh scripts/sweep_mnist_mlpnet_woodfisher.sh 0.95 1 0  #0.95: target sparsity, 1: random seed for data sampling, 0: GPU ID
    ```
        We also provivde the dumped gradients and Hessien inverse files at [here](https://drive.google.com/drive/folders/1utO0xasvMyzIRrXSHhtp_kwBR9mKVKhD?usp=sharing). You may download them and skip this step.   
 
    3.2 Run the proposed RMP + LS algorithm (CBS-S):

    ```
        sh scripts/mnist-mlpnet-backbone_layers-fullfisher_fisher-greedy_online_magperb_all_layers.sh 0.95 1e-4 10 20 0 0 test
        # 0.95: target sparsity
        # 1e-4: min impact variation for changing (\epsilon)
        # 10:   range of candidates for each weight swap(\rou)
        # 20:  τ - max failed weight swap attempts
        # 0: seed
        # 0: GPU
        # test: a flag to make the log of the experiment flush to stdin
    ```
        This experiment will dump the result at {greedy_path}, which will be requred to run CBS-U. You can check {greedy_path} at the end of the experiment log.

    Please check the explaination of the above hyperparameters in Algorithm1 of the main paper.

    3.3 Run our combinatorial weight update (CBS-U):
    ```
        sh scripts/mnist-mlpnet-backbone_layers-fullfisher_fisher-greedy_magperb-update_rm_multiple.sh 0.8 1e-4 10 20  ${greedy_result_path} 0 0 test
        # 0.8: target sparsity
        # 1e-4: min impact variation for changing (\epsilon)
        # 10:   range of candidates for each weight swap(\rou)
        # 20:  τ - max failed weight swap attempts
        # 0: seed
        # 0: GPU
        # test: a flag to make the log of the experiment flush to stdin
    ```

    To run experiment in batch, please refer scripts/run_all_cifarnet.sh

4. Run CBS on other network architectures:

    Similarly, we provides the scripts used in our paper for ResNet20, CIFARNet, MobileNet. In our paper, the reported results are based on the average of 5 runs with seeds from 0 to 4. We list the scripts for the models respectively as below. If you are interested in the ablation study experiments, you can also find their scripts in ./scripts folder.

    ```
    ## on ResNet20
    # Get the gradients and Hessian inverse matrix
    scripts/sweep_cifar10_resnet20_woodfisherblock.sh 0.8 0 0 test
    # CBS-S (RMP + LS): 
    sh scripts/cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers.sh 0.8 1e-4 10 20 1 0test 
    # Note, you can also try CBS-S using only LS. 
    sh scripts/cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers.sh  0.8 1e-4 10 20 0 0 test
    # CSB-U 
    sh scripts/cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers-upate_rm_multiple.sh 0.8 1e-4 10 20 ${greedy_path} 0 0 test

    ## on CIFARNET
    # Get the gradients and Hessian inverse matrix
    sh scripts/sweep_cifar10_cifarnet_woodfisherblock.sh 0.95 0 0 test
    # CBS-S (RMP + LS): 
    sh scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers.sh 0.95 1e-4 10 20 1 0 test
    # Note, you can also try CBS-S using only LS. 
    # sh scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers.sh 0.95 1e-4 10 20 0 0 test
    # CSB-U    
    sh scripts/cifar10-cifarnet-backbone_layers-blockwise_fisher-mag-update_rm_multiple.sh 0.95 1e-4 10 20 ${greedy_path} 0 0 test 
    
    ## on MobleNet
    # Get the gradients and Hessian inverse matrix
    sh imagenet-mobilenet-backbone_layers-mag.sh 0.95 1 0
    # CBS-S (RMP + LS): 
    sh imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_online_magperb_all_layers.sh 0.5 1e-4 10 20 0 0 test
    # Note, you can also try CBS-S using only LS. 
    sh imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers.sh 0.5 1e-4 10 20 0 0 test
    # CSB-U   
    sh scripts/imagenet-mobilenet-backbone_layers-blockwise_fisher-greedy_mag-update_rm_multiple.sh 0.5 1e-4 10 20 ${greedy_path} 0 0 test
    ``` 

## Citaton
We thanks Singh & Alistarh for sharing their code of [WoodFisher](https://github.com/IST-DASLab/WoodFisher). Our implementation is based on their code. If our CBS work is helpful to your research/project, please cite our work as below.

``` 
@InProceedings{pmlr-v162-yu22f,
  title =    {The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks},
  author =       {Yu, Xin and Serra, Thiago and Ramalingam, Srikumar and Zhe, Shandian},
  booktitle =    {Proceedings of the 39th International Conference on Machine Learning},
  pages =    {25668--25683},
  year =     {2022}
}
``` 

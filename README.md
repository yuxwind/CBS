# CBS
Official Code of [The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks](https://proceedings.mlr.press/v162/yu22f/yu22f.pdf)[ICML2022]

# Setup the enviroment: 
    conda create -n grasp python=3.6.13
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    sh setup.sh 
   
 
# Prepare data
    mkdir prob_regressor_data
    mkdir prob_regressor_results

$ Run tests on MNIST:
    4.1 Run on woodfisher firstly. This will sample data and dump their gradients to './prob_regressor_data' for future usage.
        sh scripts/sweep_mnist_mlpnet_woodfisher.sh 0.8 1 0  #0.8: target sparsity, 1: random seed for data sampling, 0: GPU ID
    
    4.2 Run the proposed RMP + LS algorithm:
    
        sh scripts/mnist-mlpnet-backbone_layers-fullfisher_fisher-greedy_magperb-update_rm_multiple.sh 0.8 1e-4 10 20 0 0 test
        # 0.8: target sparsity
        # 1e-4: min impact variation for changing (\epsilon) 
        # 10:   range of candidates for each weight swap(\rou)
        # 20:  τ - max failed weight swap attempts 
        # 0: seed
        # 0: GPU
        # test: a flag to make the log of the experiment flush to stdin
        
        this experiment will dump the results in ./prob_regressor_results/mnist-mlpnet-backbone_layers-fullfisher_fisher-greedy_magperb-update_rm_multiple/esults.train_loss_all_samples.{TIME_STAMP}/mlpnet_mnist_10000samples_10000batches_0seed.eval/
        This path will require to pass to the script for weight update.        

    Please check the explaination of the above hyperparameters in Algorithm1 of the main paper.
    
    4.3 Run our combinatorial weight update:       
        
        sh scripts/mnist-mlpnet-backbone_layers-fullfisher_fisher-greedy_magperb-update_rm_multiple.sh 0.8 1e-4 10 20 1.0 ${greedy_result_path} 0 0 test 
        # 0.8: target sparsity
        # 1e-4: min impact variation for changing (\epsilon) 
        # 10:   range of candidates for each weight swap(\rou)
        # 20:  τ - max failed weight swap attempts 
        # 0: seed
        # 0: GPU
        # test: a flag to make the log of the experiment flush to stdin
       
    To run experiment in batch, please refer scripts/run_all_cifarnet.sh
 
 # Citaton
 
@InProceedings{pmlr-v162-yu22f,
  title = 	 {The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks},
  author =       {Yu, Xin and Serra, Thiago and Ramalingam, Srikumar and Zhe, Shandian},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {25668--25683},
  year = 	 {2022}
}

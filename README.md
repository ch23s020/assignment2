# assignment2

**Wandb Report Link:-**

https://wandb.ai/ch23s020/Assignment2_CNN/reports/Assignment-2-CNN--Vmlldzo3NDYyMzE0


CS6910:- CNN o i_naturalist dataset
**Part A:-**
The data is loaded on google drive unzipped and used directly under the code under data loading stage.

If the .py file gives any challenges please refer to google notebook version also committed here

Please use the following input format for running the script and passing the hyperparameters. Feel free to choose the difeerent values of hyperparameters. from following:

from google colab:- !python partA_main.py --num_filters 32 64 --filter_size 3 3 --activation ReLU --use_batchnorm 1 --use_dropout 0 --lr 0.001 --num_epochs 10 --filter_org same --data_augmentation yes --batch_size 64 --num_neurons 128 --learning_algorithm adam

from command prompt:- python partA_main.py --num_filters 32 64 --filter_size 3 3 --activation ReLU --use_batchnorm 1 --use_dropout 0 --lr 0.001 --num_epochs 10 --filter_org same --data_augmentation yes --batch_size 64 --num_neurons 128 --learning_algorithm Adam



'hyperparameters': {
		'num_filters'                 : {'values' :[32,64,128,256]  },
		'filter_size'                 : {'values' :[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)], [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)],                                                                                         [(7,7),(7,7),(5,5),(5,5),(3,3)] }
		'activation'                  : {'values' :['ReLU', 'GELU', 'SiLU', 'Mish'] },
                'use_batchnorm'               : {'values' :['YES','NO']} ,
   	  	'use_dropout'                 : {'values' :['same', 'double', 'halve']},
     		'lr'                          : {'values' :[0.001, 0.01, 0.0001] },
    		'num_epochs'                  : {'values' :[30,60,90] },
    		'data_augmentation'           : {'values':['YES','NO']},
     		'batchsize'                   : {'values' :[16,32,64,128] },
		 num_neurons'                 : {'values' :[20,32,64,128,512,1024] },
		'learning_algorithm'          : {'values' :['adam', 'sgd'] },
  		
PartA_main_Q1 or google colab notebook contains all the code except logging test accuracy. To avoid any confusion only logging test accuracy for the best hyperparameter ran separately through differently notebook but under the same wandb project.

For Q1 Calculation part pdf file is attached as wand report are not showing the proper values. Please refer the pdf file for the same.
		
		
 		 


**Part B:-**
For Part B used Google Colab as directly using torchvision to load the models and loading the data from drive as done in part a


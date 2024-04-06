# assignment2
CS6910

Please use the following input format for running the script and passing the hyperparameters. Feel free to choose the difeerent values of hyperparameters. from following:

from google colab:- !python debug.py --num_filters 32 64 --filter_size 3 3 --activation ReLU --use_batchnorm 1 --use_dropout 0 --lr 0.001 --num_epochs 10 --filter_org same --data_augmentation yes --batch_size 64 --num_neurons 128 --learning_algorithm adam

from command prompt:- 


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
  		
 		
		
		
 		 





Command prompt input;- python main2.py --num_filters 32,64,128,256 --filter_size "(3,3),(3,3),(3,3),(3,3)" --activation ReLU --use_batchnorm 1 --use_dropout 0 --lr 0.001 --num_epochs 10 --data_augmentation yes --batch_size 64

If using google colab:- !python main2.py --num_filters "32,64,128,256" --filter_size "[(3,3),(3,3),(3,3),(3,3)]" --activation ReLU --use_batchnorm 1 --use_dropout 0 --lr 0.001 --num_epochs 10 --data_augmentation yes --batch_size 64


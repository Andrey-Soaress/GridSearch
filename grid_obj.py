from itertools import combinations_with_replacement, product
import numpy as np

class grids:

    def __init__(self):
        self.param_grids = []
        self.atual_grid = 0
        self.len_grid = 0
        self.id_model_grid = {} #as keys vão de 0 em diante

    def is_empty(self):
        if self.atual_grid == self.len_grid: return True
        
        return False

    def add_grid(self,dict_grid,identify_model):
        if identify_model not in ['NN','RF'] :
            print('No identify model setted')
            return
        self.param_grids.append(dict_grid)
        self.len_grid+=1
        self.id_model_grid[self.len_grid-1] = identify_model

    def get_grid(self):

        if not self.is_empty():
            grid_to_return = self.param_grids[self.atual_grid]
            identify = self.id_model_grid[self.atual_grid]
            self.atual_grid+=1
            return grid_to_return,identify
        else:
            return {}


def get_param_grid():
    
    qtt_dropout_layers = 2
    
    
    arq_layers_0 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)
    
    for i in combinations_with_replacement([20],r=qtt_hidden_layers):
      arq_layers_0.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i) 
    
    n = 3
      
    param_grid_0 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_0,
        'init': [ 'uniform','normal', ], 
        'batch_size':[16],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }
    
    
    arq_layers_1 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)
    
    for i in combinations_with_replacement([40,60],r=qtt_hidden_layers):
      arq_layers_1.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i) 
    
    n = 3
      
    param_grid_1 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_1,
        'init': [ 'uniform','normal', ], 
        'batch_size':[16,32],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }

    arq_layers_2 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)

    for i in combinations_with_replacement([60,80],r=qtt_hidden_layers):
      arq_layers_2.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i) 
    
    n = 3
    
    param_grid_2 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_2,
        'init': [ 'uniform','normal', ], 
        'batch_size':[16,32],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }

    arq_layers_3 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)
    
    for i in combinations_with_replacement([80,100],r=qtt_hidden_layers):
      arq_layers_3.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i)
      
    n = 3
    
    param_grid_3 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_3,
        'init': [ 'uniform','normal', ],
        'batch_size':[16,32],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }

    arq_layers_4 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)
    
    for i in combinations_with_replacement([100,120],r=qtt_hidden_layers):
      arq_layers_4.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i)
       
    n = 3
        
    param_grid_4 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_4,
        'init': [ 'uniform','normal', ], 
        'batch_size':[16,32],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }

    arq_layers_5 = []
    qtt_hidden_layers = 3
    options_layers_do = np.eye(qtt_hidden_layers)[0:qtt_dropout_layers,:].sum(axis=0)
    
    for i in combinations_with_replacement([86,90],r=qtt_hidden_layers):
      arq_layers_5.append(i)
    temporary = []
    for i in permutations(options_layers_do): temporary.append(i)
       
    n = 3
        
    param_grid_5 = {
        'n_features': [n],
        'epochs': [400],
        'layers_size': arq_layers_5,
        'init': [ 'uniform','normal', ], 
        'batch_size':[16,32],
        'optimizer':['Adam'],
        #'kr__layers_type':[['r','r','r','r']],
        'dropout_rate':[0.0,0.2,0.5],
        'learning_rate':[0.001,0.01],
        'dropout_layers':list(set(temporary))
    }
    
    param_grid_rf = {
        'n_estimators' : [100,120,160,200], 
        'criterion' : ['squared_error','poisson'], 
        'min_samples_split' : [2,3,4], 
        'min_samples_leaf' : [1,2], 
        'bootstrap' : [True,False]
    }

    #Escrever aqui os grids

    grid_to_return = grids()
    #grid_to_return.add_grid(param_grid_0)
    #grid_to_return.add_grid(param_grid_1)
    #grid_to_return.add_grid(param_grid_2)
    #grid_to_return.add_grid(param_grid_3)
    #grid_to_return.add_grid(param_grid_4)
    #grid_to_return.add_grid(param_grid_5)
    grid_to_return.add_grid(param_grid_rf,'RF')

    return grid_to_return

    

#DEPENDENCIES
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Input
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
#from itertools import combinations_with_replacement
from create_model import create_model_RNN,create_model,create_model_RF
import grid_obj
from manage_results import save_best_result
from keras.callbacks import EarlyStopping

#Data
def get_data(input_data,separator):

    df = pd.read_csv(input_data['path'],sep=separator)
    
    X,Y = df[input_data['variables']].values,df[input_data['target']].values

    return X,Y

#This function should prepare the dataset to use in LSTM NN
def prepare_data_LSTM_model(X,Y,n_in=1,n_out=1,variables): #n_in = Number of previous steps
    
    n_vars = None
    try:
        n_vars = data.shape[1]
    except IndexError:
        n_vars = 1
        
    df = pd.DataFrame(X)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{var}'+'(t-%d)' % (i)) for j,var in enumerate(variables)]
    # forecast sequence (t, t+1, ... t+n)
     cols.append(df.shift(-0))
     names += [(f'{var}'+'(t)') for j,var in enumerate(variables)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    return agg

#Estimator_and_grid
def exe_grid(X,Y,atual_grid,type_model,variables):
    
    estimator = None
    
    if type_model == "NN":
        print('Initializing pipeline...')
        estimator = KerasRegressor(build_fn=create_model,verbose=2)
        print('Pipeline creation ended!')
    elif type_model == "RF":
        estimator = create_model_RF()
        print('RF model created!')
    elif type_model == "RNN":
        estimator = KerasRegressor(build_fn=create_model_RNN,verbose=2)
    
    param_grid = atual_grid
    
    print(f'Grid : {atual_grid}\n')
    
    kfold_splits = 5

    grid = GridSearchCV(estimator=estimator,  
                        n_jobs=-1, 
                        verbose=2,
                        scoring='r2',
                        return_train_score=False,
                        cv=kfold_splits,
                        param_grid=param_grid,)


    grid_result = grid.fit(X,Y, )

    print(f"\nBest: {grid_result.best_score_}\nUsing: {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    """
    for mean,param in zip(means,params):
        print(f"{mean} with: {param}")
    """

    return grid_result.best_params_,grid_result.best_score_

def init_process(input_data,separator):
    print('Initializing process...',end='\n\n')
    
    X,Y = get_data(input_data,separator)

    # To run in Grid Search ( loading grid object )
                                                           
    param_grid_obj = grid_obj.get_param_grid()             
                                                           
    #-------------------------------------------------------
    
    next_grid = not param_grid_obj.is_empty()
    callback = EarlyStopping(monitor='loss', patience=70)
    idx = 0#idx = 1
    
    while next_grid:
        
        param_grid,type_model = param_grid_obj.get_grid()
        
        if type_model == "NN" or 'RNN' : 
            param_grid['callbacks'] = [callback]
        
        dict_result,best_score = exe_grid(X,Y,param_grid,type_model,input_data['variables'])

        next_grid = not param_grid_obj.is_empty()

        save_best_result((dict_result,best_score),input_data['label'],idx)
        idx+=1

    print("\nProcess ended!",end='\n\n')

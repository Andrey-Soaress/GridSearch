from keras.models import Sequential
from keras.layers import Dense,Input,Dropout,SimpleRNN,LSTM
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor


def create_model(n_features, layers_size, optimizer="adam", init='normal', 
                 learning_rate=0.001, dropout_layers=None, dropout_rate=0):
    
    model = Sequential()

    model.add(Input([n_features,])) 
  
    for index,layer_size in enumerate(layers_size):
        
        model.add(Dense(units=layer_size, activation='relu',kernel_initializer = init))
        if dropout_rate > 0 and dropout_layers[index] == 1:
            model.add(Dropout(dropout_rate)) #Dropout add
    
    optimizer = Adam(learning_rate = learning_rate)
    model.add(Dense(1, activation='linear', kernel_initializer = init))
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=["mean_squared_error"])
      
    print(model.summary())
    return model

def create_model_RF():
    
    rf = RandomForestRegressor()
    
    return rf


def create_model_RNN(neurons, init='uniform', optimizer="adam", 
                     dropout=0.0, rec_dropout=0.0, X_shape):
    
    model = Sequential()
    
    model.add(LSTM(neurons, input_shape=(X_shape[1],X_shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics='mean_squared_error')

  return model


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:45:26 2018

@author: dallums
"""

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
import matplotlib.pyplot as plt



def oneHot(_df, _col_list, _verbose=False):
    """
    Returns dataframe with the relevant column one-hotted
    """
    
    for _col in _col_list:
        if _verbose:
            print('Working on %s...' %_col)
        _df[_col] = _df[_col].astype(str)
        _df_col = _df[_col].str.get_dummies()
        _df = pd.concat([_df, _df_col], axis=1)
        if _verbose:
            print('Finished with %s...' %_col)
        
    return _df


def loadInitialData(_filename, _to_drop):
    """
    Load in the data from a filename; save as csv
    """
    
    # read in the csv
    dataset = read_csv(filename,  parse_dates = [['year', 'month', 'day', 'hour']], 
                       index_col=0, date_parser=lambda x: datetime.strptime(x, '%Y %m %d %H'))
    dataset.drop('No', axis=1, inplace=True)
    
    # manually specify column names
    #dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    
    # mark all NA values with 0
    #dataset['pollution'].fillna(0, inplace=True)
    dataset.fillna(0, inplace=True)
    
    # drop the first _to_drop rows
    dataset = dataset[_to_drop:]

    # save to file
    dataset.to_csv('pollution.csv')
    
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    
    # specify columns to plot
    cols_to_plot = []
    for i, col in enumerate(dataset.columns):
        if type(dataset[col].values[0]) != str:
            cols_to_plot.append(i)
    
    # plot each column
    pyplot.figure()
    for i, col in enumerate(cols_to_plot):
        	pyplot.subplot(len(cols_to_plot), 1, i+1)
        	pyplot.plot(values[:, col])
        	pyplot.title(dataset.columns[col], y=0.5, loc='right')
        	i += 1
    pyplot.show()
    
    return dataset


def series_to_supervised(data, n_in=10, n_out=1, dropnan=True):
    """
    Unrolling the time series so it can be used for supervised learning.
    n_in is the number of time steps to look back, n_out is number
    to look ahead.
    """
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg


def dataToTrainAndTest(_lookback, _lookforward, _cols_to_one_hot, _train_percentage):
    """
    After data had been pre-processed, load it into test and train sets
    """

    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    
    # one-hotting relevant columns
    dataset = oneHot(dataset, _cols_to_one_hot, True)
    dataset = dataset.drop(_cols_to_one_hot, axis=1)
    values = dataset.values
    
    # ensure all data is float
    values = values.astype('float32')
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # specify the number of lag hours
    n_hours = _lookback
    n_features = values.shape[1]
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, _lookforward)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = int(_train_percentage*len(dataset))
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    return train_X, test_X, train_y, test_y, n_hours, n_features, scaler
    
    
def buildNetwork(_train_X, _test_X, _train_y, _test_y, _hours, _features, 
                 _scaler, _epochs, _hidden_nodes, _batch_size):
    """
    Build the model and make predictions
    """

    # design network
    model = Sequential()
    model.add(LSTM(_hidden_nodes, input_shape=(_train_X.shape[1], _train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    # fit network
    history = model.fit(_train_X, _train_y, epochs=_epochs, batch_size=_batch_size, 
                        validation_data=(_test_X, _test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
    # make a prediction
    yhat = model.predict(_test_X)
    print(yhat.shape,_test_X.shape)
    _test_X = _test_X.reshape((_test_X.shape[0], _hours*_features))
    
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, _test_X[:, -(test_X.shape[2]-1):]), axis=1)
    inv_yhat = _scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    # invert scaling for actual
    _test_y = _test_y.reshape((len(_test_y), 1))
    inv_y = concatenate((_test_y, _test_X[:, -(test_X.shape[2]-1):]), axis=1)
    inv_y = _scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    return yhat, inv_yhat, _test_X, _test_y, inv_y



if __name__ == '__main__':
    filename = 'PRSA_data_2010.1.1-2014.12.31.csv' # source filename
    lookback = 1 # number of rows to lookback in time series
    lookforward = 1 # number of rows to lookforward in time series 
    epochs = 10 # training epochs
    hidden_nodes = 500
    batch_size = 72
    train_percentage = .2 # percentage of data to train on
    toDrop = 24 # initial rows to drop from dataframe
    x1 = 1000 # for plotting results - index to begin with in test set
    x2 = 1100 # for plotting results - index to end with in test set
    x1 -= lookback # for plotting consistency
    x2 -= lookback # for plotting consistency
    cols_to_one_hot = ['cbwd']
    
    dataset = loadInitialData(filename, toDrop)
    agg_data = series_to_supervised(dataset, n_in=lookback, n_out=lookforward, dropnan=True)
    train_X, test_X, train_y, test_y, hours, features, scaler = dataToTrainAndTest(lookback, 
                                                                                   lookforward, 
                                                                                   cols_to_one_hot, 
                                                                                   train_percentage)
    yhat, inv_yhat, test_X, test_y, inv_y = buildNetwork(train_X, test_X, train_y, test_y, 
                                                         hours, features, scaler, epochs, 
                                                         hidden_nodes, batch_size)
       
    plt.plot([x for x in range(len(inv_yhat[x1:x2]))], inv_yhat[x1:x2], inv_y[x1:x2])
    
    
    
    
    
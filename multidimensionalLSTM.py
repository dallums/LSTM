#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:45:26 2018

@author: dallums
"""

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

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



def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')


def loadInitialData(_filename):
    """
    Load in the data from a filename; save as csv
    """
    dataset = read_csv(filename,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv('pollution.csv')
    
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        	pyplot.subplot(len(groups), 1, i)
        	pyplot.plot(values[:, group])
        	pyplot.title(dataset.columns[group], y=0.5, loc='right')
        	i += 1
    pyplot.show()
    
    return dataset


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Unrolling the time series so it can be used for supervised learning
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


def dataToTrainAndTest():
    """
    After data had been pre-processed, load it into test and train sets
    """
    
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_hours = 3
    n_features = 8
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24 #no ten before
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
    
    
def buildNetwork(_train_X, _test_X, _train_y, _test_y, _hours, _features, _scaler, _epochs):
    """
    Build the model and make predictions
    """

    # design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(_train_X.shape[1], _train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(_train_X, _train_y, epochs=_epochs, batch_size=72, validation_data=(_test_X, _test_y), verbose=2, shuffle=False)
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
    inv_yhat = concatenate((yhat, _test_X[:, -7:]), axis=1)
    inv_yhat = _scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    _test_y = _test_y.reshape((len(_test_y), 1))
    inv_y = concatenate((_test_y, _test_X[:, -7:]), axis=1)
    inv_y = _scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    return yhat, inv_yhat, _test_X, _test_y, inv_y



if __name__ == '__main__':
    filename = 'PRSA_data_2010.1.1-2014.12.31.csv'
    epochs = 1
    dataset = loadInitialData(filename)
    agg_data = series_to_supervised(dataset, n_in=1, n_out=1, dropnan=True)
    train_X, test_X, train_y, test_y, hours, features, scaler = dataToTrainAndTest()
    yhat, inv_yhat, test_X, test_y, inv_y = buildNetwork(train_X, test_X, train_y, test_y, hours, features, scaler, epochs)
    plt.plot([x for x in range(1, 10001)], inv_yhat[:10000], inv_y[:10000])
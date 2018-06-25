#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 07:49:02 2018

@author: dallums
"""



import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler



class  DataGeneratorSeq(object):
    
    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)
        
        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)
                
            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(1,5)]
            
            self._cursor[b] = (self._cursor[b]+1)%self._prices_length
            
        return batch_data,batch_labels
    
    def unroll_batches(self):
            
        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):
            
            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels
    
    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))
     


def getData(_filename):
    """
    Takes in csv, returns df
    """ 
       
    df = pd.read_csv(os.path.join(_filename),delimiter=',',usecols=['Date','Open','High','Low','Close'])    
    df = df.sort_values('Date')
    
    return df


def splitIntoTestAndTrainAndNormalize(_df, _LSTM_hyperparameter_dict):
    """
    Split the data into a test and train set, and calculate the mid price.
    Also normalizes. Cutoff should be a percentage
    """

    # First calculate the mid prices from the highest and lowest 
    high_prices = _df['High'].as_matrix()
    low_prices = _df['Low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0
    
    cutoff = _LSTM_hyperparameter_dict['cutoff']
    length = len(_df)
    cutoffIndexFloat = cutoff*length
    cutoffIndex = int(cutoffIndexFloat)
    train_data = mid_prices[:cutoffIndex]
    test_data = mid_prices[cutoffIndex:]
    
    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data w.r.t training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    
    # Train the Scaler with training data and smooth data 
    smoothing_window_size = _LSTM_hyperparameter_dict['smoothing_window_size']
    
    for di in range(0,len(train_data)-smoothing_window_size,smoothing_window_size): #second arg was hardcoded originally at 10000. 
        # I think it just needs to be less than len(train_data but not sure)
        print(di, di+smoothing_window_size)
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # You normalize the last bit of remaining data 
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    
    # Reshape both train and test data
    train_data = train_data.reshape(-1)
    
    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    
    all_mid_data = np.concatenate([train_data,test_data],axis=0)
    
    return train_data, test_data, all_mid_data


def inputPlaceholders(_LSTM_hyperparameter_dict):
    """
    Defining placeholders for our inputs to be used in the optimazation step
    """
    
    batch_size = _LSTM_hyperparameter_dict['batch size']
    D = _LSTM_hyperparameter_dict['D']
    num_unrollings = _LSTM_hyperparameter_dict['num unrollings']
    
    tf.reset_default_graph() # This is important in case you run this multiple times
    
    train_inputs, train_outputs = [],[]
    
    # You unroll the input over time defining placeholders for each time step
    for ui in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
        train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))
        
    return train_inputs, train_outputs


def parametersOfLSTMAndRegressionLayer(_LSTM_hyperparameter_dict):
    """
    You will have three layers of LSTMs and a linear regression layer (denoted by w and b), 
    that takes the output of the last LSTM cell and output the prediction for the next time step. 
    You can use the MultiRNNCell in TensorFlow to encapsualate the three LSTMCell objects 
    you created. Additionally you can have the dropout implemented LSTM cells, as they improve 
    performance and reduce overfitting.
    """
    
    num_nodes = _LSTM_hyperparameter_dict['num nodes']
    n_layers = _LSTM_hyperparameter_dict['n_layers']
    dropout = _LSTM_hyperparameter_dict['dropout']
    
    lstm_cells = [
        tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                                state_is_tuple=True,
                                initializer= tf.contrib.layers.xavier_initializer()
                               )
        for li in range(n_layers)]
    
    drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
    ) for lstm in lstm_cells]
    drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    
    _w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
    _b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))
    
    return drop_multi_cell, multi_cell, _w, _b


def feedingLSTMOutputToRegressionLayer(_LSTM_hyperparameter_dict):
    """
    In this section, you first create TensorFlow variables (c and h) that will hold the cell state 
    and the hidden state of the LSTM. Then you transform the list of train_inputs to have a shape of 
    [num_unrollings, batch_size, D], this is needed for calculating the outputs with the tf.nn.dynamic_rnn 
    function. You then calculate the lstm outputs with the tf.nn.dynamic_rnn function and split the 
    output back to a list of num_unrolling tensors. the loss between the predictions and true stock prices.
    """

    n_layers = _LSTM_hyperparameter_dict['n_layers']
    batch_size = _LSTM_hyperparameter_dict['batch size']
    num_nodes = _LSTM_hyperparameter_dict['num nodes']
    num_unrollings = _LSTM_hyperparameter_dict['num unrollings']
    train_inputs, train_outputs = inputPlaceholders(_LSTM_hyperparameter_dict)
    drop_multi_cell, multi_cell, _w, _b = parametersOfLSTMAndRegressionLayer(_LSTM_hyperparameter_dict)
    
    # Create cell state and hidden state variables to maintain the state of the LSTM# Create 
    c, h = [],[]
    initial_state = []
    for li in range(n_layers):
      c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))
    
    # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of 
    # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)
    
    # all_outputs is [seq_length, batch_size, num_nodes]
    all_lstm_outputs, state = tf.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
        time_major = True, dtype=tf.float32)
    
    all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])
    
    all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,_w,_b)
    
    split_outputs = tf.split(all_outputs,num_unrollings,axis=0)
    
    return c, h, split_outputs, state, train_outputs, train_inputs, _w, _b, drop_multi_cell, multi_cell


def lossAndOptimizer(_LSTM_hyperparameter_dict):
    """
    Here you calculate the loss. However note that there is a unique characteristic 
    when calculating the loss. For each batch of predictions and true outputs, you calculate
    the mean squared error. And you sum (not average) all these mean squared losses together. 
    Finally you define the optimizer you're going to use to optimize the LSTM. Here you can 
    use Adam, which is a very recent and well-performing optimizer.
    """
    
    n_layers = _LSTM_hyperparameter_dict['n_layers']
    num_unrollings = _LSTM_hyperparameter_dict['num unrollings']
    c, h, split_outputs, state, train_outputs, train_inputs, _w, _b, drop_multi_cell, multi_cell = feedingLSTMOutputToRegressionLayer(_LSTM_hyperparameter_dict)
    
    # When calculating the loss you need to be careful about the exact form, because you calculate
    # loss of all the unrolled steps at the same time
    # Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps
    
    print('Defining training Loss')
    loss = 0.0
    with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+
                                 [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
      for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)
    
    print('Learning rate decay operations')
    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step,global_step + 1)
    tf_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
    tf_min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
    
    learning_rate = tf.maximum(
        tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
        tf_min_learning_rate)
    
    # Optimizer.
    print('TF Optimization operations')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v))
    
    print('\tAll done')
    
    return tf_learning_rate, tf_min_learning_rate, optimizer, loss, inc_gstep, _w, _b, drop_multi_cell, multi_cell, c, h, split_outputs, state, train_outputs, train_inputs
    
    
def predictionRelatedCalculations(_LSTM_hyperparameter_dict):
    """
    Here you define the prediction related TensorFlow operations. First define a placeholder 
    for feeding in the input (sample_inputs), then similar to the training stage, you define 
    state variables for prediction (sample_c and sample_h). Finally you calculate the prediction 
    with the tf.nn.dynamic_rnn function and then sending the output through the regression 
    layer (w and b). You also should define the reset_sample_state opeartion, that resets the 
    cell state and the hidden state of the LSTM. You should execute this operation at the start, 
    every time you make a sequence of predictions.
    """
    
    D = _LSTM_hyperparameter_dict['D']
    n_layers = _LSTM_hyperparameter_dict['n_layers']
    num_nodes = _LSTM_hyperparameter_dict['num nodes']
    tf_learning_rate, tf_min_learning_rate, optimizer, loss, inc_gstep, _w, _b, drop_multi_cell, multi_cell, c, h, split_outputs, state, train_outputs, train_inputs = lossAndOptimizer(_LSTM_hyperparameter_dict)
    
    print('Defining prediction related TF functions')
    
    sample_inputs = tf.placeholder(tf.float32, shape=[1,D])
    
    # Maintaining LSTM state for prediction stage
    sample_c, sample_h, initial_sample_state = [],[],[]
    for li in range(n_layers):
      sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))
    
    reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                                   *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])
    
    sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                       initial_state=tuple(initial_sample_state),
                                       time_major = True,
                                       dtype=tf.float32)
    
    with tf.control_dependencies([tf.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                                  [tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
      sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), _w, _b)
    
    print('\tAll done')
    
    return sample_inputs, sample_prediction, reset_sample_states, tf_learning_rate, tf_min_learning_rate, optimizer, loss, inc_gstep, _w, _b, drop_multi_cell, multi_cell, c, h, split_outputs, state, train_outputs, train_inputs

    
def runLSTM(_df, _LSTM_hyperparameter_dict):
    """
    Here you will train and predict stock price movements for several epochs and see whether the predictions get better or worse over time. You follow the following procedure.

    Define a test set of starting points (test_points_seq) on the time series to evaluate the LSTM at
    For each epoch
        For full sequence length of training data
            Unroll a set of num_unrollings batches
            Train the LSTM with the unrolled batches
        Calculate the average training loss
        For each starting point in the test set
            Update the LSTM state by iterating through the previous num_unrollings data points found before the test point
            Make predictions for n_predict_once steps continuously, using the previous prediction as the current input
            Calculate the MSE loss between the n_predict_once points predicted and the true stock prices at those time stamps
    """

    train_data, test_data, all_mid_data = splitIntoTestAndTrainAndNormalize(_df, _LSTM_hyperparameter_dict)
    #c, h, split_outputs, state, train_outputs, train_inputs = feedingLSTMOutputToRegressionLayer(_LSTM_hyperparameter_dict)
    #tf_learning_rate, tf_min_learning_rate, optimizer, loss, inc_gstep, _w, _b, drop_multi_cell, multi_cell = lossAndOptimizer(_LSTM_hyperparameter_dict)
    sample_inputs, sample_prediction, reset_sample_states, tf_learning_rate, tf_min_learning_rate, optimizer, loss, inc_gstep, _w, _b, drop_multi_cell, multi_cell, c, h, split_outputs, state, train_outputs, train_inputs = predictionRelatedCalculations(_LSTM_hyperparameter_dict)
    
    num_unrollings = _LSTM_hyperparameter_dict['num unrollings']
    batch_size = _LSTM_hyperparameter_dict['batch size']
    epochs = _LSTM_hyperparameter_dict['epochs']
    n_predict_once = _LSTM_hyperparameter_dict['n_predict_once']
    valid_summary = _LSTM_hyperparameter_dict['valid_summary']
    
    train_seq_length = train_data.size # Full length of the training data
    
    train_mse_ot = [] # Accumulate Train losses
    test_mse_ot = [] # Accumulate Test loss
    predictions_over_time = [] # Accumulate predictions
    MSE_by_epoch = [] # MSE by epoch 
    
    session = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()
    
    # Used for decaying learning rate
    loss_nondecrease_count = 0
    loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate
    
    print('Initialized')
    average_loss = 0
    
    # Define data generator
    data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings) 
    
    x_axis_seq = []
    
    # Points you start our test predictions from
    test_points_seq = np.arange(len(train_data), len(_df)-n_predict_once-1, n_predict_once).tolist() 
    
    for ep in range(epochs):       
        
        # ========================= Training =====================================
        for step in range(train_seq_length//batch_size):
            
            u_data, u_labels = data_gen.unroll_batches()
    
            feed_dict = {}
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
                feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)
            
            feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})
    
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    
            average_loss += l
        
        # ============================ Validation ==============================
        if (ep+1) % valid_summary == 0:
    
          average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))
          
          # The average loss
          if (ep+1)%valid_summary==0:
            print('Average loss at step %d: %f' % (ep+1, average_loss))
          
          train_mse_ot.append(average_loss)
                
          average_loss = 0 # reset loss
          
          predictions_seq = []
          
          mse_test_loss_seq = []
          
          # ===================== Updating State and Making Predicitons ========================
          for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []
            
            if (ep+1)-valid_summary==0:
              # Only calculate x_axis values in the first validation epoch
              x_axis=[]
            
            # Feed in the recent past behavior of stock prices
            # to make predictions from that point onwards
            for tr_i in range(w_i-num_unrollings+1,w_i-1):
              current_price = all_mid_data[tr_i]
              feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)    
              _ = session.run(sample_prediction,feed_dict=feed_dict)
            
            feed_dict = {}
            
            current_price = all_mid_data[w_i-1]
            
            feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
            
            # Make predictions for this many steps
            # Each prediction uses previous prediciton as it's current input
            for pred_i in range(n_predict_once):
    
              pred = session.run(sample_prediction,feed_dict=feed_dict)
            
              our_predictions.append(np.asscalar(pred))
            
              feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)
    
              if (ep+1)-valid_summary==0:
                # Only calculate x_axis values in the first validation epoch
                x_axis.append(w_i+pred_i)
    
              mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2
            
            session.run(reset_sample_states)
            
            predictions_seq.append(np.array(our_predictions))
            
            mse_test_loss /= n_predict_once
            mse_test_loss_seq.append(mse_test_loss)
            
            if (ep+1)-valid_summary==0:
              x_axis_seq.append(x_axis)
            
          current_test_mse = np.mean(mse_test_loss_seq)
          
          # Learning rate decay logic
          if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
              loss_nondecrease_count += 1
          else:
              loss_nondecrease_count = 0
          
          if loss_nondecrease_count > loss_nondecrease_threshold :
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')
          
          test_mse_ot.append(current_test_mse)
          print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
          MSE_by_epoch.append(mse_test_loss_seq)
          predictions_over_time.append(predictions_seq)
          print('\tFinished Predictions')
      
    return predictions_over_time, x_axis_seq, all_mid_data, MSE_by_epoch

        
def plotResults(_predictions_over_time, _x_axis_seq, _all_mid_data, _df, _LSTM_hyperparameter_dict, _MSE_by_epoch): 
    """
    Visualizing the results
    """
         
    train_data, test_data, all_mid_data = splitIntoTestAndTrainAndNormalize(_df, _LSTM_hyperparameter_dict) 
                          
    best_prediction_epoch = _MSE_by_epoch.index(min(_MSE_by_epoch)) # replace this with the epoch that you got the best results when running the plotting code
    
    plt.figure(figsize = (18,18))
    plt.subplot(2,1,1)
    plt.plot(range(_df.shape[0]),_all_mid_data,color='b')
    
    # Plotting how the predictions change over time
    # Plot older predictions with low alpha and newer predictions with high alpha
    start_alpha = 0.25
    alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(_predictions_over_time[::3]))
    for p_i,p in enumerate(_predictions_over_time[::3]):
        for xval,yval in zip(_x_axis_seq,p):
            plt.plot(xval,yval,color='r',alpha=alpha[p_i])
    
    plt.title('Evolution of Test Predictions Over Time',fontsize=18)
    plt.xticks(range(0,_df.shape[0],500),_df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.xlim(len(train_data),len(_df))
    
    plt.subplot(2,1,2)
    
    # Predicting the best test prediction you got
    plt.plot(range(_df.shape[0]),_all_mid_data,color='b')
    for xval,yval in zip(_x_axis_seq,_predictions_over_time[best_prediction_epoch]):
        plt.plot(xval,yval,color='r')
        
    plt.title('Best Test Predictions Over Time',fontsize=18)
    plt.xticks(range(0,_df.shape[0],500),_df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.xlim(len(train_data),len(_df))
    plt.show()



if __name__ == '__main__':
    filename = 'IBM.csv'
    LSTM_hyperparameter_dict = {'D': 1,
                               'cutoff': .9,
                               'smoothing_window_size': 2500,
                               'batch size': 500, # Number of samples in a batch
                               'num unrollings': 50, # Number of time steps you look into the future.
                               'D': 1, # Dimensionality of the data. Since our data is 1-D this would be 1
                               'num nodes': [200, 200, 150], # Number of hidden nodes in each layer of the deep LSTM stack we're using
                               'n_layers': 3, # number of layers, should be length of num nodes above
                               'dropout': .2, # dropout amount
                               'epochs': 30,
                               'valid_summary': 1, # Interval you make test predictions
                               'n_predict_once': 200 # Number of steps you continously predict for
                               }
    
    df = getData(filename)
    predictions_over_time, x_axis_seq, all_mid_data, MSE_by_epoch = runLSTM(df, LSTM_hyperparameter_dict)
    plotResults(predictions_over_time, x_axis_seq, all_mid_data, df, LSTM_hyperparameter_dict, MSE_by_epoch)
    
    




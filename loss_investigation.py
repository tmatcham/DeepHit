# I want to find out why the reported hazard rate is lower than expected.
# It seems that the data is not the problem
# Now want to use the data, in combination with the loss function to see if loss is minimised by the correct answer or not.

#First load the data, run it through the loss function (in batches?) with optimal parameters, figure out where its going wrong

import import_data as impt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
# import sys

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepHit import Model_DeepHit
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score, c_t_index

(x_dim), (data, time, label), (mask1, mask2, mask3) = impt.import_dataset_SYNTHETIC_CROSSING(
    norm_mode='standard')
EVAL_TIMES = [10]


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here
    return data

def get_pred():
    data_mode = 'SYNTHETIC_CROSSING'  # METABRIC, SYNTHETIC

    in_path = data_mode + '/results_TD/'
    if not os.path.exists(in_path):
        os.makedirs(in_path)

    _, num_Event, num_Category = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]

    out_itr = 0

    ##### MAIN SETTING
    OUT_ITERATION               = 1

    data_mode                   = 'SYNTHETIC_CROSSING' #METABRIC, SYNTHETIC
    seed                        = 1234

    EVAL_TIMES                  = [10] # evalution times (for C-index and Brier-Score)

    #get predictions
    in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters.txt'
    in_parser = load_logging(in_hypfile)

    ##### HYPER-PARAMETERS
    mb_size = in_parser['mb_size']

    iteration = in_parser['iteration']

    keep_prob = in_parser['keep_prob']
    lr_train = in_parser['lr_train']

    h_dim_shared = in_parser['h_dim_shared']
    h_dim_CS = in_parser['h_dim_CS']
    num_layers_shared = in_parser['num_layers_shared']
    num_layers_CS = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn = tf.nn.tanh
    else:
        print('Error!')

    initial_W = tf.contrib.layers.xavier_initializer()

    alpha = in_parser['alpha']  # for log-likelihood loss
    beta = in_parser['beta']  # for ranking loss
    gamma = in_parser['gamma']  # for RNN-prediction loss
    delta = in_parser['delta']
    parameter_name = 'a' + str('%02.0f' % (10 * alpha)) + 'b' + str('%02.0f' % (10 * beta)) + 'c' + str(
        '%02.0f' % (10 * gamma)) + 'd' + str( '%02.0f' % (10 * delta))

    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims = {'x_dim': x_dim,
                  'num_Event': num_Event,
                  'num_Category': num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings = {'h_dim_shared': h_dim_shared,
                        'h_dim_CS': h_dim_CS,
                        'num_layers_shared': num_layers_shared,
                        'num_layers_CS': num_layers_CS,
                        'active_fn': active_fn,
                        'initial_W': initial_W}

    # for out_itr in range(OUT_ITERATION):
    print('ITR: ' + str(out_itr + 1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(
        gamma) + ' d:' + str(delta) + ')')
    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data, te_data, tr_time, te_time, tr_label, te_label,
     tr_mask1, te_mask1, tr_mask2, te_mask2, tr_mask3, te_mask3) = \
        train_test_split(data, time, label, mask1, mask2, mask3, test_size=0.20, random_state=seed)

    (tr_data, va_data, tr_time, va_time, tr_label, va_label,
     tr_mask1, va_mask1, tr_mask2, va_mask2, tr_mask3, va_mask3) = \
        train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, test_size=0.20,
                         random_state=seed)

    ##### PREDICTION & EVALUATION
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### PREDICTION
    pred = model.predict(te_data)
    return pred, te_mask1, te_mask2, te_mask3, te_data, te_time, te_label
pred, te_mask1, te_mask2, te_mask3 , te_data, te_time, te_label= get_pred()

import matplotlib.pyplot as plt

ind1 = te_data[:,0] < 0
ind2 = te_data[:,0] > 0
pred1 = pred[ind1,0,:]
pred2 = pred[ind2,0,:]

F1 = np.append(0, np.cumsum(pred1[0, :-1]))
S1 = 1 - F1
H1 = - np.log(S1)
h1 = np.diff(H1)
h1 = h1[:10]

F2 = np.append(0, np.cumsum(pred2[0, :-1]))
S2 = 1 - F2
H2 = - np.log(S2)
h2 = np.diff(H2)
h2 = h2[:10]

n = 1950

xmin = [i for i in range(10)]
xmax = [i for i in range(1,11)]
F1 = np.append(0, np.cumsum(pred1[0, :-1]))
S1 = 1 - F1
H1 = - np.log(S1)
h1 = np.diff(H1)
h1 = h1[:10]

F2 = np.append(0, np.cumsum(pred2[0, :-1]))
S2 = 1 - F2
H2 = - np.log(S2)
h2 = np.diff(H2)
h2 = h2[:10]
plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', label='Z=0', alpha=1)
plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', label='Z=1', alpha=1)

for i in range(n-1):
    F1 = np.append(0, np.cumsum(pred1[i, :-1]))
    S1 = 1 - F1
    H1 = - np.log(S1)
    h1 = np.diff(H1)
    h1 = h1[:10]

    F2 = np.append(0, np.cumsum(pred2[i, :-1]))
    S2 = 1 - F2
    H2 = - np.log(S2)
    h2 = np.diff(H2)
    h2 = h2[:10]
    plt.hlines(h1, xmin, xmax, colors='b', linestyles='solid', alpha = 0.01)
    plt.hlines(h2, xmin, xmax, colors='r', linestyles='solid', alpha = 0.01)
x = [i+0.5 for i in range(10)]
ticks = [str(i) for i in range(1,11)]
plt.xticks(x, ticks, minor = False)
plt.xlabel('t')
plt.ylabel(r'$h(t\vert z)$')
plt.legend(loc ='center left')


#### Check data
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()


index = te_data[:,0] < 0
t0 = te_time[index]
e0 = te_label[index]
kmf.fit(t0,e0)
kmf.plot()


t1 = te_time[~index]
e1 = te_label[~index]
kmf.fit(t1,e1)
kmf.plot()

#Data being trained is as expected



#Now to check the loss function

import tensorflow as tf
_EPSILON = 1e-04

def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

#need to calculate what the correct predictions are supposed to be
#The hazard is 0.5 and 0.05 for one group and 0.05  and 0.5 for the other
#how does this convert to pdf?
#F(t) = 1 - S(t)
#S(t) = exp( -H(t) )
#S(t) = exp( -0.5t ) for t < 5
#F(t) = 1 - exp( -0.5(5) -0.05(t-5) t>= 5
#f(t_interval) = F(t_2) - F(t_1)
import numpy as np
#First calculate F(t) for each time point


#calculate the pmf and then the cdf?
n = 10000
h1 = 0.5
h2 = 0.05

#calculate the pmf and then the cdf?
hazard1 = [h1 for i in range(5)]
hazard1.extend([h2 for i in range(5)])

hazard2 = [h2 for i in range(5)]
hazard2.extend([h1 for i in range(5)])

pi1 = [hazard1[0]]
for i in range(1, 10):
        pi1.append(hazard1[i] * (1- np.sum(pi1)))
pi2 = [hazard2[0]]
for i in range(1, 10):
        pi2.append(hazard2[i] * (1 - np.sum(pi2)))

F1 = np.cumsum(pi1)
F1 = np.append(F1, F1[-1]+(1 - F1[-1])/2)
F1 = np.append(F1, 1)
F1 = np.append(0, F1)
f1 = np.diff(F1)

F2 = np.cumsum(pi2)
F2 = np.append(F2, F2[-1]+(1 - F2[-1])/2)
F2 = np.append(F2, 1)
F2 = np.append(0, F2)
f2 = np.diff(F2)

import matplotlib.pyplot as plt
plt.plot([1-k for k in F1])

truth = np.zeros((pred.shape[0], 1, 12))
for i in range(pred.shape[0]):
    if te_data[i,0] > 0:
        truth[i,0,:] = f2
    else:
        truth[i,0,:] = f1

def loss(output):
    k = te_label
    y_out = output
    fc_mask1 = te_mask1
    fc_mask2 = te_mask2
    I_1 = tf.sign(k)
    # for uncenosred: log P(T=t,K=k|x)
    tmp1 = tf.reduce_sum(tf.reduce_sum(fc_mask1 * y_out, reduction_indices=2), reduction_indices=1,
                         keep_dims=True)
    tmp1 = I_1 * log(tmp1)
    # for censored: log \sum P(T>t|x)
    tmp2 = tf.reduce_sum(tf.reduce_sum(np.expand_dims(fc_mask2,1) * y_out, reduction_indices=2), reduction_indices = 1,
                         keep_dims=True)
    tmp2 = (1. - I_1) * log(1.-tmp2)
    LOSS_1 = - tf.reduce_mean(tmp1 + 1.0 * tmp2)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    result = sess.run(LOSS_1)

    print(result) # print 20
    sess.close()
    return

loss(pred)
loss(truth)

# LOSS FUNCTION - RANKING LOSS 1
def loss_Ranking(output):
    k = te_label
    fc_mask2 = te_mask2.astype(np.float32)
    t = te_time.astype(np.float32)
    _, num_Event, num_Category = np.shape(te_mask1)

    sigma1 = tf.constant(0.1, dtype=tf.float32)

    eta = []
    for e in range(num_Event):
        one_vector = tf.ones_like(t, dtype=tf.float32)
        I_2 = tf.cast(tf.equal(k, e + 1), dtype=tf.float32)  # indicator for event
        I_2 = tf.diag(tf.squeeze(I_2))
        tmp_e = tf.reshape(tf.slice(output, [0, e, 0], [-1, 1, -1]),
                           [-1, num_Category])  # event specific joint prob.

        R = tf.matmul(tmp_e, tf.transpose(fc_mask2))  # no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
        R = tf.matmul(one_vector, tf.transpose(diag_R)) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = tf.transpose(R)  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = tf.nn.relu(
            tf.sign(tf.matmul(one_vector, tf.transpose(t)) - tf.matmul(t, tf.transpose(one_vector))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = tf.matmul(I_2, T)  # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), reduction_indices=1, keep_dims=True)

        eta.append(tmp_eta)
    eta = tf.stack(eta, axis=1)  # stack referenced on subjects
    eta = tf.reduce_mean(tf.reshape(eta, [-1, num_Event]), reduction_indices=1, keep_dims=True)

    LOSS_2 = tf.reduce_mean(tf.reduce_mean(eta, reduction_indices=0),reduction_indices =0)  # mean over num_Events

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    result = sess.run(R)

    print(result) # print 20
    sess.close()

loss_Ranking(pred)
loss_Ranking(truth.astype('float32'))

### LOSS-FUNCTION 2 -- Haz Ranking loss
def loss_Ranking(output):
    k = te_label
    fc_mask3 = te_mask3.astype(np.float32)
    t = te_time.astype(np.float32)
    _, num_Event, num_Category = np.shape(te_mask1)

    sigma1 = tf.constant(1, dtype=tf.float32)

    eta = []
    for e in range(num_Event):
        one_vector = tf.ones_like(t, dtype=tf.float32)
        zero_vector = tf.zeros_like(t, dtype=tf.float32)
        I_2 = tf.cast(tf.equal(k, e + 1), dtype=tf.float32)  # indicator for event (0 if censor, 1 if event)
        I_2 = tf.diag(tf.squeeze(I_2))  #Diagonal indicator function
        tmp_e = tf.reshape(tf.slice(output, [0, e, 0], [-1, 1, -1]), [-1, num_Category])  # event specific joint prob.
        tmp_e1 = tf.cumsum(tmp_e, 1)
        tmp_e1 = tf.concat((zero_vector, tmp_e1), axis = 1)
        tmp_e1 = tf.slice(tmp_e1, [0,0], [-1, num_Category])
        tmp_e1 = 1.0 - tmp_e1
        tmp_e1 = tf.math.maximum(tmp_e1, 0.0)
        tmp_e2 = div(tmp_e, tmp_e1) #hazard for each individual at each time step

        R = tf.matmul(tmp_e2, tf.transpose(fc_mask3))  # no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
        R = tf.matmul(one_vector, tf.transpose(diag_R)) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = tf.transpose(R)  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = tf.nn.relu(
            tf.sign(tf.matmul(one_vector, tf.transpose(t)) - tf.matmul(t, tf.transpose(one_vector))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = tf.matmul(I_2, T)  # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), reduction_indices=1, keep_dims=True)

        eta.append(tmp_eta)
    eta = tf.stack(eta, axis=1)  # stack referenced on subjects
    eta = tf.reduce_mean(tf.reshape(eta, [-1, num_Event]), reduction_indices=0, keep_dims=True)

    LOSS_2 = tf.reduce_sum(eta)  # sum over num_Events

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    result = sess.run(R)

    print(result) # print 20
    sess.close()
    return result

loss_Ranking(pred)
loss_Ranking(truth.astype('float32'))





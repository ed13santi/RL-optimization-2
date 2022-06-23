###################################################################################
###################################################################################
# Control task rates using actions between 0 and 1 (proportion of new tasks to 
# be accepted for each task branch). Reward is the sum of utilities if constraints 
# are not exceeded, and 0 if constraints are exceeded.
###################################################################################
###################################################################################



## Import packages

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import pickle
import os 
from memory_profiler import profile
from keras import backend as Kbackend
import gc


## Functions of objective and constraint

def sqrt_func(a, b, x):
    return a * x**0.5 + b

def der_sqrt_func(a, x):
    if x == 0:
        return 0.5 * a / (0.0001**0.5)
    return 0.5 * a / (x**0.5)

def log_func(a, b, x):
    return np.log(np.dot(a, x) + 1) + b

def der_log_func(a, x, b):
    return a / (np.dot(a, x) + 1) + b
    
def lin_constr(A, C, x):
    return C- np.matmul(A,x)
    
def obj_func(x, local_objs):
    s = 0
    for (x_i, f) in zip(x, local_objs):
        s += f(x_i)
    return s

def obj_func_der(x, local_ders, N):
    out = np.zeros((N,))
    for i in range(N):
        out[i] = local_ders[i](x[i])
    return out
    




class CstmLoss(keras.losses.Loss):
  def __init__(self, weigths):
    super().__init__()  
    self.weights = weigths

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    weights = tf.convert_to_tensor(self.weights)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true - weights * y_pred, axis=-1)





class Q_approx:
    def __init__(self, no_tasks, learning_rate, max_x):
        self.state_size = no_tasks
        self.lr = learning_rate
        self.max_x = max_x
        self.no_actions = self._no_actions(max_x)
        self.NN = self._init_NN(self.state_size)
        self.target_NN = self._init_targetNN(self.state_size)

    def _no_actions(self, max_x):
        max_x_actions = max_x + 1
        return np.prod(max_x_actions)

    def _init_NN(self, state_size):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(state_size,), activation="relu", name="layer1"))
        model.add(layers.Dense(16, activation="relu", name="layer2"))
        model.add(layers.Dense(self.no_actions, activation="linear", name="layer3"))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _init_targetNN(self, state_size):
        model = self._init_NN(state_size)
        weights = self.NN.get_weights()
        model.set_weights(weights)
        return model

    def _pred1(self, NN, inp):
        return NN(np.expand_dims(inp, axis = 0))

    def get_action(self, s, epsilon):
        if np.random.uniform() - epsilon:
            Qs = np.random.rand(self.no_actions)
        else:
            Qs = self._pred1(self.NN, s).numpy()[0]
        Qs = np.reshape(Qs, self.max_x+1)
        return np.squeeze(np.array(np.nonzero(Qs == np.max(Qs))))

    def train(self, replay_mem, gamma, batch_size, N): 
        if len(replay_mem) >= 50: #don't train if there are less than 50 samples
            SARS = random.sample(replay_mem, batch_size)
            S = np.array([el[:N] for el in SARS])
            A = np.array([el[N:2*N] for el in SARS]).astype(int)
            R = np.array([el[2*N] for el in SARS])
            S_new = np.array([el[2*N+1:] for el in SARS])
            predicted = self.NN(S).numpy()
            predicted_next = self.target_NN(S_new).numpy()
            for i in range(predicted.shape[0]):
                tmp = np.reshape(predicted[i], self.no_actions)
                tmp2 = tmp.copy()
                tmp2[A[i]] = R[i] + gamma * np.max(predicted_next[i])
                predicted2 = predicted
                predicted2[i] = tmp2.flatten()
            self.NN.train_on_batch(S, predicted2) 

        




def g_tplus1(x_new, constr_A, C):
    return C - np.matmul(constr_A, x_new)

# return new state (new number of processed tasks is deterministic while number of incoming tasks is sampled from a uniform distribution)
def state_trans(a, max_x):
    return np.rint(max_x * a)

def reward(x, local_objs, constr_A, C, N):
    if np.any(np.multiply(constr_A, x) - C > 0):
        return 0

    local_rs = [local_objs[i](x[i]) for i in range(N)]
    return sum(local_rs)
    

def global2i_state(state, i, N):
    xplusm_i = np.array([state[i]])
    lambdas = state[N:]
    return np.concatenate((xplusm_i, lambdas))

def locals2global_state(local_states):
    xplusm = [el[0] for el in local_states]
    lamb = local_states[0][1:]
    return np.concatenate(xplusm,lamb)

def indexer(arr, indexes):
    out = arr
    for i in indexes:
        out = out[i]
    return out


def main():
    ## Set parameters

    # utility functions are a * x^0.5 + b
    a_vect = [1,1]
    b_vect = [0,0]

    N = len(a_vect) # number of tasks

    constr_A = np.array([[1,0],
                        [0,1]])
    C = np.array([2,2])
    type_obj = 0 # 0 is sqrt root, 1 is log
        
        
    gamma = 0.99
    n_episodes = 100000
    episode_length = 1000
    replay_mem = deque(maxlen = 5000)

    update_every = 50
    batch_size = 100

    storage_path = "/rds/general/user/eds17/home/RL-optimization-2/centralisedDQNsaturated/"
    os.makedirs(os.path.dirname(storage_path + "replay_mem.pkl"), exist_ok=True)

    local_objs = []
    for a,b in zip(a_vect, b_vect):
        if type_obj == 0:
            local_objs.append(lambda x, a=a, b=b: sqrt_func(a,b,x))
        else:
            local_objs.append(lambda x, a=a, b=b: log_func(a,b,x))
            
    local_ders = []
    for a,b in zip(a_vect, b_vect):
        if type_obj == 0:
            local_ders.append(lambda x, a=a: der_sqrt_func(a,x))
        else:
            local_ders.append(lambda x, a=a: der_log_func(a,x))

    # calculate max possible x for each task
    tmp_constr_A = copy.deepcopy(constr_A).astype(float)
    tmp_constr_A[tmp_constr_A == 0] = 0.00001  # avoid divide by 0
    max_x = np.amin(C/tmp_constr_A, axis=0).astype(int)


    a_std = 0.01 # standard deviation of noise applied to action, not sure what value it should be
    learning_rate = 0.001

    epsilon = 1
    epsilon_decay = 0.999

    for episode in range(n_episodes):
        if episode != 0:
            Kbackend.clear_session()
            del q_approximator
            gc.collect()
            
         # create list of actors and critics 
        q_approximator = Q_approx(N, learning_rate, max_x)

        if episode != 0:
            #print("NEW EPISODE!!!!")
            filehandler =  open(storage_path + "replay_mem", 'rb') 
            replay_mem = pickle.load(filehandler)
            q_approximator.NN.load_weights(storage_path + "q")
            q_approximator.target_NN.load_weights(storage_path + "qTrgt")


        state = np.random.uniform(np.zeros((N,)), 2*max_x+2, (N,))
        state = state.astype(int)
        rs = [] 
        us = []
        state_record = []
        x_record = []

        if episode % 10 == 0:
            print(episode)

        for it in range(episode_length):
            # interaction with environment
            action = q_approximator.get_action(state, epsilon)
            new_state = state_trans(action, max_x)
            r = reward(new_state, local_objs, constr_A, C, N)
            rs.append(r)
            state_record.append(new_state)
            x_record.append(new_state)
            replay_mem.append(np.concatenate([state, action, [r], new_state]))
            state = new_state

            # update actors and critics
            if it % update_every == 0 and len(replay_mem) >= batch_size: 
                # train Q network
                q_approximator.train(replay_mem, gamma, batch_size, N)

                # update target networks
                weights = q_approximator.NN.get_weights()
                target_weights = q_approximator.target_NN.get_weights()
                new_weights = [0.1*el1+0.9*el2 for el1,el2 in zip(weights, target_weights)]
                q_approximator.target_NN.set_weights(new_weights)

            epsilon *= epsilon_decay


                        
        filehandler =  open(storage_path + "replay_mem", 'wb') 
        pickle.dump(replay_mem, filehandler)
        os.makedirs(os.path.dirname(storage_path + "rs_{}".format(episode)), exist_ok=True)
        filehandler =  open(storage_path + "rs_{}".format(episode), 'wb')
        pickle.dump(rs, filehandler)
        os.makedirs(os.path.dirname(storage_path + "us_{}".format(episode)), exist_ok=True)
        filehandler =  open(storage_path + "us_{}".format(episode), 'wb')
        pickle.dump(us, filehandler)
        os.makedirs(os.path.dirname(storage_path + "state_{}".format(episode)), exist_ok=True)
        filehandler =  open(storage_path + "state_{}".format(episode), 'wb')
        pickle.dump(state_record, filehandler)
        os.makedirs(os.path.dirname(storage_path + "xs_{}".format(episode)), exist_ok=True)
        filehandler =  open(storage_path + "xs_{}".format(episode), 'wb')
        pickle.dump(state, filehandler)
        q_approximator.NN.save_weights(storage_path + "q")
        q_approximator.target_NN.save_weights(storage_path + "qTrgt")


if __name__ == '__main__':
    main()

## Import packages

from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
import copy
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import pickle
import os 
import sys
import tracemalloc
from memory_profiler import profile
import tensorflow.keras.backend as Kbackend
import gc

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#disable_eager_execution()


## Set parameters

# utility functions are a * x^0.5 + b
a_vect = [1,1]
b_vect = [0,0]

N = len(a_vect) # number of tasks

constr_A = np.array([[1,0],
                     [0,1]], dtype="float32")
constr_A_tf = tf.convert_to_tensor(constr_A)

K = np.shape(constr_A)[0] # number of resources
K_tf = tf.convert_to_tensor(K)

C = np.array([2,2], dtype="float32")
C_tf = tf.convert_to_tensor(C)

type_obj = 0 # 0 is sqrt root, 1 is log

stop_iter = 10000
delta_x = 0.01
delta_l = 0.01

x_init = np.zeros((N,))
lambda_init = np.zeros((K,))

loc_state_size = K + 1



## Functions of objective and constraint

def sqrt_func(a, b, x):
    return a * x**0.5 + b

def der_sqrt_func(a, x):
    if x == 0:
        return 0.5 * a / (0.0001**0.5)
    return 0.5 * a / (x**0.5)

def log_func(a, b, x):
    return np.log(np.dot(a, x) + 1) + b

def der_log_func(a, x):
    return a / (np.dot(a, x) + 1) + b
    
def lin_constr(A, C, x):
    return C- np.matmul(A,x)

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

def local_ders_tf(x):
    tmpList = []
    for i in range(N):
        tmpList.append(local_ders[i](x[i]))
    return tf.squeeze(tf.stack(tmpList, axis=0))

    
def obj_func(x):
    s = 0
    for (x_i, f) in zip(x, local_objs):
        s += f(x_i)
    return s

def obj_func_der(x):
    out = np.zeros((N,))
    for i in range(N):
        out[i] = local_ders[i](x[i])
    return out
    
constr = (lambda x, A=constr_A, C=C: lin_constr(A, C, x))




class cstmLoss(keras.losses.Loss):
  def __init__(self, weigths):
    super().__init__()  
    self.weights = weigths

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    weights = tf.convert_to_tensor(self.weights)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true - weights * y_pred, axis=-1)





class Actor:
    def __init__(self, alpha_theta):        
        self.learning_rate = alpha_theta
        self.NN = self._init_NN()
        self.target_NN = self._init_targetNN()

    def _init_NN(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(loc_state_size,), activation="relu", name="layer1"))
        model.add(layers.Dense(16, activation="relu", name="layer2"))
        model.add(layers.Dense(1, activation="sigmoid", name="layer3"))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _init_targetNN(self):
        model = self._init_NN()
        weights = self.NN.get_weights()
        model.set_weights(weights)
        return model

    def _pred1(self, NN, inp):
        return NN(tf.expand_dims(inp, axis=0))

    def get_action(self, s):
        return self._pred1(self.NN, s)[0]

    def get_action_arr(self, states):
        return np.squeeze(np.array(self.NN(states)), axis=1)

    @profile
    def update(self, state_arr, action_arr, criticNN, alpha_theta):
        with tf.GradientTape() as tape:
            inputs = tf.concat((state_arr, tf.expand_dims(action_arr, axis=1)), axis=1)
            Qs = criticNN(inputs)
            wrt_vars = tf.expand_dims(inputs[:,-1], axis=1)
        batch_jac = tape.batch_jacobian(Qs, wrt_vars)
        self.NN.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha_theta), loss=cstmLoss(tf.squeeze(batch_jac, axis=2)))
        self.NN.train_on_batch(state_arr, tf.zeros(tf.shape(state_arr)[0],))
        gc.collect()
        Kbackend.clear_session()

    def update_target(self):
        actor_weights = self.NN.get_weights()
        target_actor_weights = self.target_NN.get_weights()
        new_weights = [0.1*el1+0.9*el2 for el1,el2 in zip(actor_weights, target_actor_weights)]
        self.target_NN.set_weights(new_weights)


class Critic:
    def __init__(self, alpha_theta):
        self.learning_rate = alpha_theta
        self.NN = self._init_NN()
        self.target_NN = self._init_targetNN()

    def _init_NN(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(loc_state_size+1,), activation="relu", name="layer1"))
        model.add(layers.Dense(16, activation="relu", name="layer2"))
        model.add(layers.Dense(1, activation="sigmoid", name="layer3"))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _init_targetNN(self):
        model = self._init_NN()
        weights = self.NN.get_weights()
        model.set_weights(weights)
        return model

    def _pred1(self, NN, inp):
        return NN(np.expand_dims(inp, axis = 0))[0]

    def Q_sa_trgt(self,states,actorNN):
        actions = actorNN(states)
        inputs = tf.concat((states,actions), axis=1)
        return self.target_NN(inputs)
    
    @profile
    def update(self, s_arr, a_arr, y_arr):
        inputs = tf.concat((s_arr,tf.expand_dims(a_arr, axis=1)), axis=1)
        self.NN.train_on_batch(inputs, y_arr)

    def update_target(self):
        critic_weights = self.NN.get_weights()
        target_critic_weights = self.target_NN.get_weights()
        new_weights = [0.1*el1+0.9*el2 for el1,el2 in zip(critic_weights, target_critic_weights)]
        self.target_NN.set_weights(new_weights)
        



def g_tplus1(x_new, C_tf, constr_A_tf):
    return C_tf - tf.squeeze(tf.matmul(constr_A_tf, tf.expand_dims(x_new,1)), axis=1)

# return new state (new number of processed tasks is deterministic while number of incoming tasks is sampled from a uniform distribution)
def state_trans(s,a, C_tf, constr_A_tf, max_x):
    # features of state
    x_plus_m = s[:N]
    lambdas = s[N:]

    # get x(t+1) + m(t+1)
    x_new_float = tf.math.multiply(x_plus_m, tf.squeeze(a))
    x_new = tf.zeros((N,))
    x_new_floor = tf.floor(x_new_float)
    decimals = x_new_float - x_new_floor
    decision_vars = np.random.uniform(0,1,(N,))
    addToFloor = 0.5 + 0.5 * tf.sign(tf.maximum(decimals - decision_vars, 0)) 
    x_new = x_new + addToFloor

    m_new = tf.floor(tf.random.uniform((N,), np.zeros((N,)), max_x+1))

    # get lambda(t+1)
    tmp_lambda = lambdas - delta_l * g_tplus1(x_new, C_tf, constr_A_tf)
    lambdas_new = tf.maximum(tf.ones((N,)), tmp_lambda)

    return tf.concat(values=(x_new + m_new, lambdas_new), axis=0), x_new

def rewards_parallel(global_state, xs):
    lambdas = global_state[N:]
    dUidxis = local_ders_tf(xs)
    lambdas_ais_prods = tf.squeeze(tf.matmul(tf.expand_dims(lambdas, axis=0), constr_A_tf), axis=0)
    return - tf.math.abs(dUidxis - lambdas_ais_prods)

def global2i_states(state):
    lambdas = state[N:]
    local_states = [tf.concat((state[i]*tf.ones((1,)), lambdas), axis=0) for i in range(N)]
    return tf.stack(local_states, axis=0)

def locals2global_state(local_states):
    xplusm = [el[0] for el in local_states]
    lamb = local_states[0][1:]
    return np.concatenate(xplusm,lamb)

def indexer(arr, indexes):
    out = arr
    for i in indexes:
        out = out[i]
    return out
    
@profile
def main():    
    gamma = 0.99
    n_episodes = 20
    episode_length = 1000
    replay_mem = [tf.zeros((0,4+2*K)) for _ in range(N)]

    update_every = 50
    batch_size = 100
    max_replay_mem_size = 5000

    storage_path = "/rds/general/user/eds17/home/optRLFiles/"
    os.makedirs(os.path.dirname(storage_path + "replay_mem.pkl"), exist_ok=True)



    


    # calculate max possible x for each task
    tmp_constr_A = copy.deepcopy(constr_A).astype(float)
    tmp_constr_A[tmp_constr_A == 0] = 0.00001  # avoid divide by 0
    max_x = np.amin(C/tmp_constr_A, axis=0).astype(int)


    a_std = 0.01 # not sure what this should be
    alpha_theta = 0.001
    alpha_thetaCritic = 0.001

    # create list of actors and critics 
    actors = [Actor(alpha_theta) for _ in range(N)]
    critics = [Critic(alpha_thetaCritic) for _ in range(N)]

    for episode in range(n_episodes):
        state = tf.concat((tf.random.uniform((N,), np.zeros((N,)), 2*max_x+2), tf.zeros((K,))), axis=0)
        state = tf.floor(state)
        rs = [] 

        if episode % 10 == 0:
            print(episode)

        for it in range(episode_length):
            # interaction with environment
            local_states = global2i_states(state)
            local_actions_lst = [actors[i].get_action(local_states[i]) for i in range(N)]
            local_actions = tf.stack(local_actions_lst, axis=0)
            noisy_actions = tf.random.normal(shape=tf.shape(local_actions), mean=local_actions, stddev=a_std)
            new_state, xs = state_trans(state, noisy_actions, C_tf, constr_A_tf, max_x)
            r = rewards_parallel(new_state, xs)
            rs.append(r)
            new_local_states = global2i_states(new_state)
            for i in range(N):
                new_entry = tf.concat((local_states[i], noisy_actions[i]*tf.ones((1,)), r[i]*tf.ones((1,)), new_local_states[i]), axis=0)
                replay_mem[i] = tf.concat((replay_mem[i], tf.expand_dims(new_entry, axis=0)), axis=0)
                replay_mem_len = min(episode * episode_length + it + 1, max_replay_mem_size + 1)
                if replay_mem_len > max_replay_mem_size:
                    replay_mem[i] = replay_mem[i][1:]
                    replay_mem_len = max_replay_mem_size
            state = new_state

            # update actors and critics
            if (it + 1) % update_every == 0 and episode * episode_length + it + 1 >= batch_size: 
                for i in range(N):
                    # extract <S,A,R,S'> samples
                    choices = tf.constant(random.choices(range(replay_mem_len), k=batch_size), shape=(batch_size,1))
                    batch = tf.gather_nd(replay_mem[i], choices)
                    state_arr = batch[:,:1+K]
                    action_arr = batch[:,1+K]
                    reward_arr = batch[:,2+K]
                    state_new_arr = batch[:,3+K:]

                    # calculate Q values
                    q_next_arr = critics[i].Q_sa_trgt(state_new_arr, actors[i].target_NN) 

                    # calculate delta values
                    y_arr = reward_arr + gamma * q_next_arr

                    # update actor and critic parameters
                    critics[i].update(state_arr, action_arr, y_arr)
                    actors[i].update(state_arr, action_arr, critics[i].NN, alpha_theta)

                    # update target networks
                    critics[i].update_target()
                    actors[i].update_target()

                    actor_weights = actors[i].NN.get_weights()
                    target_actor_weights = actors[i].target_NN.get_weights()
                    new_weights = [0.1*el1+0.9*el2 for el1,el2 in zip(actor_weights, target_actor_weights)]
                    actors[i].target_NN.set_weights(new_weights)
                        
        filehandler =  open(storage_path + "replay_mem", 'wb') 
        pickle.dump(replay_mem, filehandler)
        os.makedirs(os.path.dirname(storage_path + "rs_{}".format(episode)), exist_ok=True)
        filehandler =  open(storage_path + "rs_{}".format(episode), 'wb')
        pickle.dump(rs, filehandler)
        for i in range(N):
            actors[i].NN.save(storage_path + "actor_{}".format(i))
            actors[i].target_NN.save(storage_path + "actorTrgt_{}".format(i))
            critics[i].NN.save(storage_path + "critics_{}".format(i))
            critics[i].target_NN.save(storage_path + "criticsTrgt_{}".format(i))


if __name__ == '__main__':
    main()

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





class Actor:
    def __init__(self, loc_state_size, alpha_theta):
        self.alpha_theta = alpha_theta
        self.NN = self._init_NN(loc_state_size)
        self.target_NN = self._init_targetNN(loc_state_size)

    def _init_NN(self, loc_state_size):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(loc_state_size,), activation="relu", name="layer1"))
        model.add(layers.Dense(16, activation="relu", name="layer2"))
        model.add(layers.Dense(1, activation="sigmoid", name="layer3"))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.alpha_theta), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _init_targetNN(self, loc_state_size):
        model = self._init_NN(loc_state_size)
        weights = self.NN.get_weights()
        model.set_weights(weights)
        return model

    def _pred1(self, NN, inp):
        return NN(np.expand_dims(inp, axis = 0))[0]

    def get_action(self, s):
        return self._pred1(self.NN, s).numpy()[0]

    def get_action_arr(self, states):
        return np.squeeze(np.array(self.NN(states)), axis=1)

    @profile
    def update(self, state_arr, action_arr, criticNN):
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(np.concatenate((state_arr, np.expand_dims(action_arr, axis=1)), axis=1))
            Qs = criticNN(inputs)
            wrt_vars = tf.expand_dims(inputs[:,-1], axis=1)
        batch_jac = tape.batch_jacobian(Qs, wrt_vars).numpy()
        self.NN.compile(optimizer=keras.optimizers.SGD(learning_rate=self.alpha_theta), loss=CstmLoss(np.squeeze(batch_jac, axis=2)))
        self.NN.train_on_batch(state_arr, np.zeros(np.shape(state_arr)[0],))


class Critic:
    def __init__(self, loc_state_size, alpha_thetaCritic):
        self.alpha_thetaCritic = alpha_thetaCritic
        self.NN = self._init_NN(loc_state_size)
        self.target_NN = self._init_targetNN(loc_state_size)

    def _init_NN(self, loc_state_size):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(loc_state_size+1,), activation="relu", name="layer1"))
        model.add(layers.Dense(16, activation="relu", name="layer2"))
        model.add(layers.Dense(1, activation="sigmoid", name="layer3"))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.alpha_thetaCritic), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _init_targetNN(self, loc_state_size):
        model = self._init_NN(loc_state_size)
        weights = self.NN.get_weights()
        model.set_weights(weights)
        return model

    def _pred1(self, NN, inp):
        return NN(np.expand_dims(inp, axis = 0))[0]

    def Q_sa_trgt(self,states,actorNN):
        actions = actorNN(states).numpy()
        inputs = np.concatenate((states,actions), axis=1)
        return self.target_NN(inputs).numpy()

    @profile
    def update(self, s_arr, a_arr, y_arr):
        inputs = np.concatenate((s_arr,np.expand_dims(a_arr, axis=1)), axis=1)
        self.NN.train_on_batch(inputs, y_arr)
        




def g_tplus1(x_new, constr_A, C):
    return C - np.matmul(constr_A, x_new)

# return new state (new number of processed tasks is deterministic while number of incoming tasks is sampled from a uniform distribution)
def state_trans(s, a, N, max_x, delta_l, constr_A, C):
    # features of state
    x_plus_m = s[:N]
    lambdas = s[N:]

    # get x(t+1) + m(t+1)
    x_new_float = x_plus_m * a
    x_new = np.zeros((N,))
    decimals = x_new_float - np.floor(x_new_float)
    for i, (x, randomVar, dec) in enumerate(zip(x_new_float, np.random.uniform(np.zeros(N,), np.ones(N,), (N,)), decimals)):
        if randomVar < dec:
            x_new[i] = np.floor(x) + 1
        else:
            x_new[i] = np.floor(x)

    m_new = np.random.uniform(np.zeros((N,)), max_x+1, (N,)).astype(int)

    # get lambda(t+1)
    lambdas_new = np.maximum(np.ones((N,)), lambdas - delta_l * g_tplus1(x_new, constr_A, C))

    return np.concatenate((x_new + m_new, lambdas_new)), x_new

def reward_i(global_state, x_i, i, local_ders, constr_A, N):
    lambdas = global_state[N:]
    dUidxi = local_ders[i](x_i)
    lambdas_ais_prod = np.dot(lambdas, constr_A[:][i])
    return - abs(dUidxi - lambdas_ais_prod)

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


@profile
def main():

    ## Set parameters

    # utility functions are a * x^0.5 + b
    a_vect = [1,1]
    b_vect = [0,0]

    N = len(a_vect) # number of tasks

    constr_A = np.array([[1,0],
                        [0,1]])

    K = np.shape(constr_A)[0] # number of resources

    C = np.array([2,2])

    type_obj = 0 # 0 is sqrt root, 1 is log

    stop_iter = 10000
    delta_x = 0.01
    delta_l = 0.01

    x_init = np.zeros((N,))
    lambda_init = np.zeros((K,))

    
    constr = (lambda x, A=constr_A, C=C: lin_constr(A, C, x))
        
        
    gamma = 0.99
    n_episodes = 100000
    episode_length = 1000
    replay_mem = [deque(maxlen = 5000) for _ in range(N)]

    update_every = 50
    batch_size = 100

    storage_path = "/rds/general/user/eds17/home/optRLFiles/"
    storage_path = "./optRLFiles/"
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

    loc_state_size = K + 1

    # calculate max possible x for each task
    tmp_constr_A = copy.deepcopy(constr_A).astype(float)
    tmp_constr_A[tmp_constr_A == 0] = 0.00001  # avoid divide by 0
    max_x = np.amin(C/tmp_constr_A, axis=0).astype(int)


    a_std = 0.01 # not sure what this should be
    alpha_theta = 0.001
    alpha_thetaCritic = 0.001

    for episode in range(n_episodes):
        if episode != 0:
            Kbackend.clear_session()
            del actors
            del critics
            gc.collect()
            
         # create list of actors and critics 
        actors = [Actor(loc_state_size, alpha_theta) for _ in range(N)]
        critics = [Critic(loc_state_size, alpha_thetaCritic) for _ in range(N)]

        if episode != 0:
            print("NEW EPISODE!!!!")
            filehandler =  open(storage_path + "replay_mem", 'rb') 
            replay_mem = pickle.load(filehandler)
            for i in range(N):
                actors[i].NN.load_weights(storage_path + "actor_{}".format(i))
                actors[i].target_NN.load_weights(storage_path + "actorTrgt_{}".format(i))
                critics[i].NN.load_weights(storage_path + "critics_{}".format(i))
                critics[i].target_NN.load_weights(storage_path + "criticsTrgt_{}".format(i))


        state = np.concatenate((np.random.uniform(np.zeros((N,)), 2*max_x+2, (N,)), np.zeros((K,))))
        state = state.astype(int)
        rs = [] 

        if episode % 10 == 0:
            print(episode)

        for it in range(episode_length):
            # interaction with environment
            local_states = [global2i_state(state, i, N) for i in range(N)]
            local_actions = [actors[i].get_action(s_i) for i,s_i in enumerate(local_states)]
            noisy_actions = np.random.normal(loc=local_actions, scale = a_std)
            new_state, xs = state_trans(state, noisy_actions, N, max_x, delta_l, constr_A, C)
            r = [reward_i(new_state, x, i, local_ders, constr_A, N) for i,x in enumerate(xs)]
            rs.append(r)
            new_local_states = [global2i_state(new_state, i, N) for i in range(N)]
            for i in range(N):
                replay_mem[i].append(np.concatenate([local_states[i], [noisy_actions[i]], [r[i]], new_local_states[i]]))
            state = new_state

            # update actors and critics
            if it % update_every == 0 and len(replay_mem[0]) >= batch_size: 
                for i in range(N):
                    # extract <S,A,R,S'> samples
                    batch = np.array(random.sample(replay_mem[i], batch_size))
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
                    actors[i].update(state_arr, action_arr, critics[i].NN)

                    # update target networks
                    critic_weights = critics[i].NN.get_weights()
                    target_critic_weights = critics[i].target_NN.get_weights()
                    new_weights = [0.1*el1+0.9*el2 for el1,el2 in zip(critic_weights, target_critic_weights)]
                    critics[i].target_NN.set_weights(new_weights)

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
            actors[i].NN.save_weights(storage_path + "actor_{}".format(i))
            actors[i].target_NN.save_weights(storage_path + "actorTrgt_{}".format(i))
            critics[i].NN.save_weights(storage_path + "critics_{}".format(i))
            critics[i].target_NN.save_weights(storage_path + "criticsTrgt_{}".format(i))


if __name__ == '__main__':
    main()

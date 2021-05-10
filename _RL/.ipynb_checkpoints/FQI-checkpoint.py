### Adapt from ###
from _util import *
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import collections
import collections
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"



tf.keras.backend.set_floatx('float64')


### for FQE, success parallel, but overhead is very large [because many NN but few for each iter]
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class MLPNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[64,64], 
                 activation='relu', 
                 name='mlp_network', mirrored_strategy = None):
        super().__init__(name=name)
        
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.mirrored_strategy = mirrored_strategy
        tf.random.set_seed(42)
        # defining layers
#         with self.mirrored_strategy.scope():
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        self.out = tf.keras.layers.Dense(units=num_actions, activation=None)

    def call(self, state):
        # MLPNetwork(); predict()
        net = tf.cast(state, tf.float64)
        for dense in self.dense_layers:
            net = dense(net)
        out = self.out(net)
        ### for Ohio, it must be negative. But not necessary for the other
        # out = -tf.math.log(1.001 + tf.exp(out))
        return out

""" hyper-parameters
one step of fitting:
    decay_steps -> Adam
    lr -> Adam
    max_epoch -> self.model.fit
outter iter:
    max_iter
    eps
"""

"""
batch RL
deterministic policies <- Q-based
model is the Q-network
"""
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class FQI(object):
    def __init__(self, num_actions=5, init_states = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99, gpu_number = 0
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, tau = None
                 , validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001):
        ### === network ===
        self.num_A = num_actions
        self.hiddens = hiddens
        self.activation = activation
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.init_states = init_states
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        self.gpu_number = gpu_number
        
        self.tau = tau
#         tf.config.experimental.set_memory_growth(gpus[gpu_number], True)
        
        ### model, optimizer, loss, callbacks ###
        #with self.mirrored_strategy.scope():
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.model = MLPNetwork(self.num_A, hiddens, activation 
                                    #, mirrored_strategy = self.mirrored_strategy
                                   )
            self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                        lr, decay_steps=decay_steps, decay_rate=1))
            self.model.compile(loss= "mse" #'huber_loss'
                               , optimizer=self.optimizer, metrics=['mse'])
        
        self.callbacks = []
        """
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        """
        if validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss'
                                                               , patience = es_patience)]        

        
    def train(self, trajs, train_freq = 100, verbose=0, nn_verbose = 0, validation_freq = 1
             , path = None, save_freq = 10):
        with tf.device('/gpu:' + str(self.gpu_number)):
            """
            form the target value (use observed i.o. the max) -> fit 
            """
            self.trajs = trajs
            states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
            self.nn_verbose = nn_verbose
            # FQE
            """ ??? """
            old_targets = rewards / (1 - self.gamma)
            # https://keras.rstudio.com/reference/fit.html
            self.model.fit(states, old_targets, 
                           batch_size=self.batch_size, 
                           epochs=self.max_epoch, 
                           verbose= self.nn_verbose,
                           validation_split=self.validation_split,
                           validation_freq = validation_freq, 
                           callbacks=self.callbacks)
            ###############################################################
            for iteration in range(self.max_iter):
                q_next_states = self.model.predict(next_states)            
                targets = rewards + self.gamma * np.max(q_next_states, 1)
                ## model targets
                """ interesting. pay attention to below!!!
                """
                _targets = self.model.predict(states)
                _targets[range(len(actions)), actions.astype(int)] = targets
                self.model.fit(states, _targets, 
                               batch_size=self.batch_size, 
                               epochs=self.max_epoch, 
                               verbose = self.nn_verbose,
                               validation_split=self.validation_split,
                               validation_freq = validation_freq, 
                               callbacks=self.callbacks)


                target_diff = change_rate(old_targets, targets)
                self.target_diffs.append(target_diff)

                if verbose >= 1 and iteration % train_freq == 0:
                    print('----- FQI (training) iteration: {}, target_diff = {:.3f}'.format(iteration, target_diff, '-----'))
                if target_diff < self.eps:
                    break

                old_targets = targets.copy()
                if path is not None and iteration % save_freq == 0 and save_freq > 0:
                    self.model.save_weights(path)
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################            
    """ self.model.predict
    Q values? or their action probabilities???????????

    """
    def Q_func(self, states, actions = None):
        with tf.device('/gpu:' + str(self.gpu_number)):
            states = tf.cast(states, tf.float64)
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
    #         states = (states - self.mean_S) / self.std_S
            if actions is not None:
                return np.squeeze(select_each_row(self.model(states), actions.astype(int)))
            else:
                return self.model(states)

    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.amax(self.Q_func(states), axis=1)
    
    
    def A_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if actions is not None:
            return np.squeeze(self.Q_func(states, actions.astype(int))) - self.V_func(states)
        else:
            return transpose(transpose(self.Q_func(states)) - self.V_func(states)) # transpose so that to subtract V from Q in batch.     
        
    def init_state_value(self, init_states = None, trajs = None, idx=0):
        """ TODO: Check definitions. 
        """
        if init_states is None:
            states = np.array([traj[idx][0] for traj in self.trajs])
        else:
            states = init_states
        return self.V_func(states) # len-n    
    """ NOTE: deterministic. for triply robust (off-policy learning)    
    """
    
    def get_A(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.argmax(self.Q_func(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        probs = np.zeros((len(states), self.num_A))
        probs[range(len(states)), optimal_actions] = 1
        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
        
    def sample_A(self, states):
        if self.tau is not None:
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
            Qs = self.Q_func(states)
            logit = np.exp(Qs / self.tau)
            probs = logit / np.sum(logit, 1)[:, np.newaxis]
            As = [np.random.choice(self.num_A, size = 1, p = aa)[0] for aa in probs]
            return As
        else:
            return self.get_A(states)


    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class weight_policies():
    def __init__(self, pi1, pi2, w = 0.5):
        self.pi1 = pi1
        self.pi2 = pi2
        self.w = w # the weight of pi1
        
    def get_A(self, S):
        A_1 = self.pi1.sample_A(S)
        A_2 = self.pi2.sample_A(S)
        choice = np.random.binomial(n = 1, p = self.w, size = len(A_1))
        A = A_1 * choice + A_2 * (1 - choice)
        A = A.astype(np.int)
        return A

    def sample_A(self, S):
        return self.get_A(S)
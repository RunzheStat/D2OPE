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
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.multioutput import MultiOutputRegressor


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class MLPNetwork(tf.keras.Model):
    ''' weights = self.model(S_r, A_r, S_t)    
    Input for `VisitationRatioModel()`; NN parameterization
    '''
    def __init__(self, num_actions, 
                 hiddens= 64, S_dim = 15, 
                 activation='relu', gpu_number = 0, 
                 name='mlp_network', mirrored_strategy = None):
        super(MLPNetwork, self).__init__()
        self.num_actions = num_actions
        self.gpu_number = gpu_number
        self.hidden_dims = hiddens
        self.seed = 42
        
        self.deeper = 1
        
        self.input_dim = S_dim 
        
        self.reset()
        
    def reset(self):
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.input_shape2 = [self.hidden_dims, self.hidden_dims]
            self.input_shape3 = [self.hidden_dims, self.num_actions]

            self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")

            self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")

            self.w21 = self.xavier_var_creator(self.input_shape2, name = "w21")
            self.b21 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b21")

            self.w22 = self.xavier_var_creator(self.input_shape2, name = "w22")
            self.b22 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b22")

            if self.deeper:
                self.w23 = self.xavier_var_creator(self.input_shape2, name = "w23")
                self.b23 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b23")
                self.w24 = self.xavier_var_creator(self.input_shape2, name = "w24")
                self.b24 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b24")

            self.w3 = self.xavier_var_creator(self.input_shape3, name = "w3")
            self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64), name = "b3")

    def xavier_var_creator(self, input_shape, name = "w3"):
        tf.random.set_seed(self.seed)
        
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0) #/ 5
        
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var

    def call(self, state):
        """
        inputs are concatenations of S, A, S_t, A_t = [r,r,t]
        """
        z = tf.cast(state, tf.float64)
        h1 = tf.nn.leaky_relu(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        
        h2 = tf.nn.relu(tf.matmul(h2, self.w21) + self.b21)
        h2 = tf.nn.relu(tf.matmul(h2, self.w22) + self.b22)
        
        if self.deeper:
            h2 = tf.nn.relu(tf.matmul(h2, self.w23) + self.b23)
            h2 = tf.nn.relu(tf.matmul(h2, self.w24) + self.b24)
        
        out = (tf.matmul(h2, self.w3) + self.b3) #/10
        # out = - tf.math.log(1.01 + tf.exp(out))
        return out

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class FQE(object):
    def __init__(self, policy = None, gamma=0.99, num_actions=4, init_states = None
                 , gpu_number = 0, init_Q_ratio = 1.0
                 , use_RF = 0, max_iter=200, eps=0.001
                 , hiddens= 256, nn_verbose = 0, es_patience=5, lr=5e-4, batch_size=64, max_epoch=20000
                 , max_depth = 50, n_estimators = 1000, min_samples_leaf = 10
                 ):
        self.policy = policy
        self.seed = 42
        ### === network ===
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.init_states = init_states
        S_dim = len(np.atleast_1d(init_states[0]))
        ### === optimization ===
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.validation_split = 0.2
        ###### 
        self.init_Q_ratio = init_Q_ratio
        self.gpu_number = gpu_number
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        self.nn_verbose = nn_verbose
        ###
        self.use_RF = use_RF
        ######################################################################################################
        if self.use_RF:
            self.model = RF(max_depth = max_depth, n_estimators = n_estimators, min_samples_leaf = min_samples_leaf
                            , n_jobs = -1, verbose = 0, random_state = 0) 

        else:
            self.model = MLPNetwork(num_actions = num_actions, hiddens = hiddens
                                    , gpu_number = gpu_number, S_dim = S_dim)
            self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    1e-3, decay_steps=10000, decay_rate = 0.99) #  * 10
                                                      , clipnorm = 1, clipvalue = 0.5)
            self.model.compile(loss= "mse" #'huber_loss'
                               , optimizer=self.optimizer, metrics=['mse'])
        ######################################################################################################
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor= 'val_loss' #'loss' #  'val_loss' #
                                                           , patience = es_patience)]        

        
    def train(self, trajs, test_freq = 10, verbose=0):
        """
        form the target value (use observed i.o. the max) -> fit 
        """
        validation_freq = 1
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        
        states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        idx = np.random.permutation(len(states))
        states, actions, rewards, next_states = states[idx], actions[idx], rewards[idx], next_states[idx]

        action_init_states = self.policy.get_A(self.init_states)
        _actions = self.policy.get_A(next_states)
        
        self.u_S = np.mean(states, 0)
        self.std_S = np.std(states, 0)
        
        states = (states - self.u_S) / self.std_S
        next_states = (next_states - self.u_S) / self.std_S
        
        # FQE
        old_targets = rewards / (1 - self.gamma) * self.init_Q_ratio #(3 / 4) #(2 / 3)
            
        ###################################################
        if self.use_RF:
            pred = states
            _targets = tf.repeat(old_targets[:, np.newaxis], self.num_actions, axis = 1)
            if pred.ndim == 1:
                pred = pred.reshape((-1, 1))
            self.model.fit(pred, _targets)
        else:
            with tf.device('/gpu:' + str(self.gpu_number)):
                pred = states
                _targets = tf.repeat(old_targets[:, np.newaxis], self.num_actions, axis = 1)
                self.model.fit(pred, _targets, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epoch, 
                       verbose= self.nn_verbose,
                       validation_split=self.validation_split,
                       validation_freq = validation_freq, 
                       callbacks=self.callbacks)   
        ###############################################################

        if not self.use_RF:
            self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    self.lr, decay_steps = self.decay_steps, decay_rate = 0.99
                                                    )
                                                      , clipnorm = 1, clipvalue = 0.5)
            self.model.compile(loss= "mse" #'huber_loss'
                               , optimizer=self.optimizer, metrics=['mse'])

        for iteration in range(self.max_iter):

            ############# Model the target ##############
            if states.ndim == 1:
                states = states.reshape((-1, 1))
                next_states = next_states.reshape((-1, 1))
            q_next_states = self.model.predict(next_states)
            targets = rewards + self.gamma * q_next_states[range(len(_actions)), _actions]
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions.astype(int)] = targets
            pred = states     
            ###################################################
            if self.use_RF:
                if pred.ndim == 1:
                    pred = pred.reshape((-1, 1))
                self.model.fit(pred, _targets)
            else:
                with tf.device('/gpu:' + str(self.gpu_number)):
                    self.model.fit(pred, _targets, 
                           batch_size=self.batch_size, 
                           epochs=self.max_epoch, 
                           verbose = self.nn_verbose,
                           validation_split=self.validation_split,
                           validation_freq = validation_freq, 
                           callbacks=self.callbacks)
            ###################################################
            ##########################################################################################
            target_diff = change_rate(old_targets, targets)
            if verbose >= 1 and iteration % test_freq == 0:
                # est_value = mean(self.init_state_value(self.init_states))
                values = self.Q_func(self.init_states, action_init_states)
                est_value = np.mean(values)
                printG("Value of FQE iter {} = {:.2f} with diff = {:.4f} and std_init_Q = {:.1f}".format(iteration, est_value, target_diff
                                                                                          , std(values)))
            ######## Stopping Creteria Here ########
            if target_diff < self.eps:
                break
            old_targets = targets.copy()
##########################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    def V_func(self, states):
        
        if len(states.shape) == 1:
            states = np.expand_dims(states, 1)
            
        _actions = self.policy.get_A(states)
        return self.Q_func(states, _actions)
    
    def Q_func(self, states, actions = None):
        if self.use_RF:
            if len(states.shape) == 1:
                states = np.expand_dims(states, 1)
            states = (states - self.u_S) / self.std_S
            if actions is not None:
                return np.squeeze(select_each_row(self.model.predict(states), actions.astype(int)))
            else:
                return self.model.predict(states)
        else:
            with tf.device('/gpu:' + str(self.gpu_number)):
                if len(states.shape) == 1:
                    states = np.expand_dims(states, 1)
                states = (states - self.u_S) / self.std_S
                if actions is not None:
                    return np.squeeze(select_each_row(self.model.predict(states), actions.astype(int)))
                else:
                    return self.model.predict(states)
        
    def init_state_value(self, init_states = None, trajs = None, idx=0):
        if init_states is None:
            states = np.array([traj[idx][0] for traj in trajs])
        else:
            states = init_states
        return self.V_func(states) # len-n
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


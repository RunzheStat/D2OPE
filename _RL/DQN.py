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
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

tf.keras.backend.set_floatx('float32')

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class DQN_online():
    def __init__(self, path_suffic = "1002_", dqn = None, env = None
                , gamma = 0.8, sd_G = 3, epsilon = 0.05
                , T = 100):
        self.FQI_path = 'target_policies/' + path_suffic
        self.recorders = {}
        self.plotlosses = PlotLosses(groups = {'value': ['value']})
        self.disc_values = np.zeros(100000)
        self.mean_values = np.zeros(100000)
        self.T = T
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.dqn = dqn
        self.e = 0
        
        os.mkdir(self.FQI_path)
    def train_one_epoch(self, save_freq = 10):
        i = self.e
        T = self.T
        observation = self.env.reset(T)
        
        self.recorders[i] = {"state" : [], "action" : [], "reward" : []}
        observation = self.env.reset(T)
        while True:
            if rbin(n = 1, p = self.epsilon):
                action = np.random.choice(5, p = np.repeat(1, 5) / 5)
            else:
                action = self.dqn.get_A(observation)[0]
            observation_, reward, done = self.env.online_step(action)
            self.dqn.replay_buffer.add([observation, action, reward, observation_])

            self.recorders[i]["state"].append(observation)
            self.recorders[i]["action"].append(action)
            self.recorders[i]["reward"].append(reward)
            self.dqn.fit_one_step(print_freq = 5, verbose = 0, nn_verbose = 0)

            if done:
                running_rewards = np.array([r for r in self.recorders[i]["reward"]])
                mean_reward = mean(running_rewards)
                gammas = arr([self.gamma ** t for t in range(len(running_rewards))])
                self.disc_values[i] = np.sum(running_rewards * gammas)
                self.mean_values[i] = mean_reward
                if i >= 10 and i % 10 == 0:
                    if i >= 100:
                        v = np.mean(self.mean_values[(i - 100):i])
                    elif i >= 50:
                        v = np.mean(self.mean_values[(i - 50):i])
                    else:
                        v = np.mean(self.mean_values[(i - 10):i])
                    print("episode:", i, "  reward:", v)
                    self.plotlosses.update({'value': v})
                    self.plotlosses.send()
                elif i < 10:
                    v = mean_reward
                    print("episode:", i, "  reward:", v)
                    self.plotlosses.update({'value': v})
                    self.plotlosses.send()
                break

            observation = observation_
        if i > 0 and i % save_freq == 0:
            self.dqn.model.save_weights(self.FQI_path + "/iter" + str(i))
        self.e += 1
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class MLPNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[64,64], 
                 activation='relu', 
                 name='mlp_network', mirrored_strategy = None):
        super().__init__(name=name)
        tf.keras.backend.set_floatx('float32')

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
#         net = tf.cast(state, tf.float64)
        net = tf.cast(state, tf.float32)
        for dense in self.dense_layers:
            net = dense(net)
        out = self.out(net)
        ### for Ohio, it must be negative. But not necessary for the other
        # out = -tf.math.log(1.001 + tf.exp(out))
        return out
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class DQN(object):
    def __init__(self, num_actions=5, init_trajs = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99, gpu_number = 0
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, 
                 validation_split=0.2):
        ### === network ===
        self.num_A = num_actions
        self.hiddens = hiddens
        self.activation = activation
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.replay_buffer = SimpleReplayBuffer(init_trajs)
        ### === optimization ===
        self.batch_size = batch_size
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        
        self.target_diffs = []
        self.values = []
        self.gpu_number = gpu_number
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.model = MLPNetwork(self.num_A, hiddens, activation 
                                    #, mirrored_strategy = self.mirrored_strategy
                                   )
            self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                        lr, decay_steps=decay_steps, decay_rate=1))
            self.model.compile(loss= "mse" #'huber_loss'
                               , optimizer=self.optimizer, metrics=['mse'])
        self.callbacks = []
        
        self.validation_freq = 1
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            old_targets = rewards / (1 - self.gamma)
            self.model.fit(states, old_targets, 
                           batch_size=self.batch_size, 
                           epochs= 1, 
                           verbose= 2,
                           validation_split=self.validation_split,
                           validation_freq = self.validation_freq, 
                           callbacks=self.callbacks)
    
    def fit_one_step(self, print_freq = 5, verbose = 0, nn_verbose = 0):
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            q_next_states = self.model.predict(next_states)
            targets = rewards + self.gamma * np.max(q_next_states, 1)
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions.astype(int)] = targets
            with tf.GradientTape() as tape:
                pred_targets = self.model(states)
                loss = tf.keras.losses.MSE(_targets, pred_targets)

            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            if verbose >= 1 and iteration % print_freq == 0:
                print('----- FQI (training) iteration: {}, target_diff = {:.3f}, values = {:.3f}'.format(iteration, target_diff, targets.mean())
                    , '-----')

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    def Q_func(self, states, actions = None):
        with tf.device('/gpu:' + str(self.gpu_number)):
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
        return self.get_A(states)

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


class SimpleReplayBuffer():
    # trajs: a list of trajs, each traj is a list of 4-tuple (S, A, R, S') [[[S, A, R, S']]]
    # aim: [arr of S, arr of A, arr of SS]
    # his original code? much easier?
    # not complex enough. online, etc.
    def __init__(self, trajs):
        self.seed = 42
        self.states = np.array([item[0] for traj in trajs for item in traj])
        self.actions = np.array([item[1] for traj in trajs for item in traj])
        self.rewards = np.array([item[2] for traj in trajs for item in traj])
        self.next_states = np.array([item[3] for traj in trajs for item in traj])
        self.N, self.S_dims = self.states.shape
        
    def add(self, SARS):
        self.states = np.append(self.states, SARS[0][np.newaxis, :], axis = 0)
        self.next_states = np.append(self.next_states, SARS[3][np.newaxis, :], axis = 0)
        self.actions = np.append(self.actions, SARS[1])
        self.rewards = np.append(self.rewards, SARS[2])

        
    def sample(self, batch_size):
        
        np.random.seed(self.seed)
        self.seed += 1
        idx = np.random.choice(self.N, batch_size, replace = False)
        return [self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]]
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


class DQN_gym_online():
    def __init__(self, path_suffic = "1002_", dqn = None, env = None
                , gamma = 0.8, sd_G = 3, epsilon = 0.05, n_actions = 2
                , T = 100):
        self.FQI_path = 'target_policies/' + path_suffic
        self.recorders = {}
        self.plotlosses = PlotLosses(groups = {'value': ['value']})
        self.disc_values = np.zeros(100000)
        self.mean_values = np.zeros(100000)
        self.T = T
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.dqn = dqn
        self.e = 0
        self.n_actions = n_actions
        try:
            os.mkdir(self.FQI_path)
        except:
            pass
    def train_one_epoch(self, save_freq = 10):
        i = self.e
        """ not useful here """
        observation = self.env.reset()

        self.recorders[i] = {"state" : [], "action" : [], "reward" : []}
        observation = self.env.reset()
        t = 0
        while True or t <= self.T:
            if rbin(n = 1, p = self.epsilon):
                action = np.random.choice(self.n_actions, p = np.repeat(1, self.n_actions) / self.n_actions)
            else:
                action = self.dqn.get_A(observation)[0]
            observation_, reward, done, info = self.env.step(action)
            self.dqn.replay_buffer.add([observation, action, reward, observation_])

            self.recorders[i]["state"].append(observation)
            self.recorders[i]["action"].append(action)
            self.recorders[i]["reward"].append(reward)
            self.dqn.fit_one_step(print_freq = 5, verbose = 0, nn_verbose = 0)

            if done:
                running_rewards = np.array([r for r in self.recorders[i]["reward"]])
                mean_reward = mean(running_rewards)
                gammas = arr([self.gamma ** t for t in range(len(running_rewards))])
                self.disc_values[i] = np.sum(running_rewards * gammas)
                self.mean_values[i] = mean_reward
                if i >= 10 and i % 10 == 0:
                    if i >= 100:
                        v = np.mean(self.disc_values[(i - 100):i])
                    elif i >= 50:
                        v = np.mean(self.disc_values[(i - 50):i])
                    else:
                        v = np.mean(self.disc_values[(i - 10):i])
                    print("episode:", i, "  reward:", v)
                    self.plotlosses.update({'value': v})
                    self.plotlosses.send()
                elif i < 10:
                    v = self.disc_values[i]
                    print("episode:", i, "  reward:", v)
                    self.plotlosses.update({'value': v})
                    self.plotlosses.send()
                break

            observation = observation_
            t += 1
        if i > 0 and i % save_freq == 0:
            self.dqn.model.save_weights(self.FQI_path + "/iter" + str(i))
        self.e += 1

        
############################################################################################################################################################################################################################################################################################################################################################

class DQN_gym():
    def __init__(self, num_states, num_actions, hidden_units, gamma, batch_size = 64
                 , lr = 0.01, max_experiences = 10000, min_experiences = 100
                 , gpu_number = 7):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.gpu_number = gpu_number
        with tf.device('/gpu:' + str(gpu_number)):
            self.model = MLPNetwork(num_actions, hidden_units
                                    #, mirrored_strategy = self.mirrored_strategy
                                   )
        #self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
    def predict(self, inputs):
        with tf.device('/gpu:' + str(self.gpu_number)):
            # inputs = tf.cast(inputs, tf.float32)
            return self.model(np.atleast_2d(inputs)) # .astype('float32')
    def train(self, TargetNet):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if len(self.experience['s']) < self.min_experiences:
                return 0
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])
            rewards = np.asarray([self.experience['r'][i] for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            dones = np.asarray([self.experience['done'][i] for i in ids])
            value_next = np.max(TargetNet.predict(states_next), axis=1)
            """ gamma here """
            actual_values = np.where(dones, rewards, rewards + self.gamma*value_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss
    def get_action(self, states, epsilon = 0):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            As = np.argmax(self.predict(np.atleast_2d(states)), 1)
            As = np.squeeze(As)
            return As #np.argmax(self.predict(np.atleast_2d(states))[0])
    def get_A(self, states):
        return self.get_action(states)
    def sample_A(self, states):
        return self.get_A(states)
    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        
        probs = np.zeros((len(states), self.num_actions))
        probs[range(len(states)), optimal_actions] = 1

        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


############################################################################################################################################################################################################################################################################################################################################################
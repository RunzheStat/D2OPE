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
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

     
import _cartpole as cartpole
reload(cartpole)
env = cartpole.CartPoleEnv(e_max = 1000)
       
            
class softmax_policy():
    def __init__(self, pi, tau):
        self.pi = pi
        self.tau = tau
    def sample_A(self, S):
        probs = self.get_A_prob(S)
        c = probs.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        As = (u < c).argmax(axis=1)
        As = np.squeeze(As)
        return As
    def get_A(self, S): 
        return self.sample_A(S)
    def get_A_prob(self, S):
        S = np.atleast_2d(S)
        Qs = self.pi.model(S).numpy()
        probs = softmax(Qs / self.tau, axis = 1)
        return probs

    
class GymEval():
    def __init__(self, random_pi = False):
        self.random_pi = random_pi
        self.seed = 42
    
    def eval_policy(self, pi, gamma, init_states = None, rep = 1000):
        rewards = self.simu_trajs_para(pi, rep = rep, T = 500
                   , burn_in = None, init_states = init_states
                       , return_rewards = True)
        Vs = []
        for i in range(rep):
            V = sum(r * gamma ** t for t, r in enumerate(rewards[i]))
            Vs.append(V)
        V_true = np.mean(Vs)
        std_true_value = np.std(Vs) / np.sqrt(len(Vs))
        printR("value = {:.4f} with std {:.4f}".format(V_true, std_true_value))
        return V_true
    
    def simu_trajs(self, pi, rep = 100, T = 1000
                   , burn_in = None, init_states = None, return_rewards = False):
        ######## 
        envs = [cartpole.CartPoleEnv(e_max = 1000) for i in range(rep)]
        Ss = randn(rep, 4)
        for i in range(rep):
            S = envs[i].reset()
            if init_states is not None:
                init_S = init_states[i]
                envs[i].state = init_S
                Ss[i] = init_S
            else:
                Ss[i] = S
        trajs = [[] for i in range(rep)]
        rewards = [[] for i in range(rep)]
        ############
        for t in range(T):
            np.random.seed(self.seed)
            self.seed += 1
            if t * 2 % T == 0:
                print("simu {}% DONE!".format(str(t / T * 100)))
            if self.random_pi:
                As = pi.sample_A(Ss)
            else:
                As = pi.get_A(Ss)
            for i in range(rep):
                SS, reward, done, _ = envs[i].step(As[i])
                SARS = [Ss[i].copy(), As[i], reward, SS.copy()]
                Ss[i] = SS
                trajs[i].append(SARS)
                rewards[i].append(reward)
        ############
        if return_rewards:
            return rewards
        if burn_in is not None:
            trajs = [traj[burn_in:] for traj in trajs]
        return trajs
    
    def simu_trajs_para(self, pi, rep = 100, T = 1000
                   , burn_in = None, init_states = None, return_rewards = False):
        ######## 
        env = cartpole.CartPoleEnv(e_max = 1000)
        Ss = env.reset_multiple(rep)
        Ss = Ss.T
        if init_states is not None:
            env.states = init_states.T
            Ss = init_states
        trajs = [[] for i in range(rep)]
        rewards = [[] for i in range(rep)]
        ############
        for t in range(T):
            np.random.seed(self.seed)
            self.seed += 1
            if t * 2 % T == 0:
                print("simu {}% DONE!".format(str(t / T * 100)))
            if self.random_pi:
                As = pi.sample_A(Ss)
            else:
                As = pi.get_A(Ss)
            SSs, Rs, _, _ = env.step_multiple(As)
            for i in range(rep):
                SARS = [Ss[i].copy(), As[i], Rs[i], SSs.T[i].copy()]
                trajs[i].append(SARS)
                rewards[i].append(Rs[i])
            Ss = SSs.T
        ############
        if return_rewards:
            return rewards
        if burn_in is not None:
            trajs = [traj[burn_in:] for traj in trajs]
        return trajs

    
    def get_init_S_from_trajs(self, trajs, n_init = 1000):
        np.random.seed(42)
        states = np.array([item[0] for traj in trajs for item in traj])
        return states[np.random.choice(len(states), n_init)]    
############################################################################################################################################################################################################################################################################################################################################################
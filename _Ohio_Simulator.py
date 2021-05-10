#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
TODO: allow online interating. take a step.
"""

""" Usage
1. evaluation
    1. (multi-seeds) simulate a dataset with the behaviour policy, OPE the target policy.
    2. evaluate in the simulator (multi-seeds)
2. offline RL
    1. (multi-seeds) simulate a dataset with the behaviour policy, learn a policy (ours or competing), and evaluate in the simulator (multi-seeds)
"""
# import os, sys
# package_path = os.path.dirname(os.path.abspath(os.getcwd()))
# sys.path.insert(0, package_path + "/test_func")
# from _util_TRPO import *
from _util import *
import operator
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


########################################################################################################################################################################################################################################################################################################################################################################################


class OhioSimulator():
    def __init__(self, sd_G = 3, T = 20, N = 10, T_burnin = 1000
                 , behav = None
                 , lag = 4, noiseless = False, equal_A = False):
        # the following parameters will not change with the LM fitting
        self.lag = lag
        self.noiseless = noiseless
        self.behav = behav
        ######
#         if self.lag == 1:
#             self.CONST = 0
#         else:
        self.CONST = 39.03
        
        ####################
        self.init_u_G = 162  
        self.init_sd_G = 60
        ####################
        self.p_D, self.u_D, self.sd_D = 0.17, 44.4, 35.5
        self.p_E, self.u_E, self.sd_E = 0.05, 4.9, 1.04
        self.range_a = [0, 1, 2, 3, 4]
        # left to right: t-4, .. , t-1
        if equal_A:
            self.p_A = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            self.p_A = [0.805, 0.084, 0.072, 0.029, 0.010] # new discritization
        
        if self.lag == 1:
            ## debug only
            if self.noiseless:
                self.coefficients = [.8, 0, 0, -8]
                #self.coefficients = [.8, 0, 0, -4]
            else:
                self.coefficients = [.8, 0.23, -3.489, 0]
        elif self.lag == 4:
            if self.noiseless:
                self.coefficients = [-0.008     ,  0.106     , -0.481     ,  1.171  # glucose
                  , 0.00     ,  -0.00     ,  0.0     ,  0      # diet
                  , 0.00     , 0     , 0     , 0  # exercise
                  , -0.30402253, -2.02343638, -0.3310525 , -0.43941028] # action
            else:
                self.coefficients = [-0.008     ,  0.106     , -0.481     ,  1.171  # glucose
                  , 0.008     ,  -0.004     ,  0.08      ,  0.23      # diet
                  , 0.009     , -1.542     , 3.097     , -3.489  # exercise
                  #, 0, 0, 0, 0]
                  , -0.30402253, -2.02343638, -0.3310525 , -0.43941028] # action

        self.tran_mat = np.expand_dims(arr(self.coefficients), 0)
        ####################
        self.sd_G = sd_G
        self.T, self.N = T, N
        self.seed = 42
        self.T_burnin = T_burnin
        
    def Glucose2Reward(self, gls):
        low_gl, high_gl = 80, 140
        rewards = np.select([gls >= high_gl, gls <= low_gl, np.multiply(low_gl < gls, gls < high_gl)]
          , [-(gls - high_gl) ** 1.35 / 30, -(low_gl - gls) ** 2 / 30, 0])
        return rewards
############################################################################################################################################
############################################################################################################################################

    def init_MDPs(self, seed = 0, N = None): 
        """ Randomly initialize 
        1. G_t [0,..., 4]
        1. the other state variable: random.
        2. errors for G_t
        3. when to take how many diets/exercises [matters?]
        
        where T varies, seed is diff; dominated by init several states, and hence values are not monotone.
        
        self.T_burnin should only be here
        """
        np.random.seed(seed)
        if N is None:
            N = self.N
        T = self.T + self.T_burnin
        obs = np.zeros((3, T, N))  # [Gi, D, Ex]
        e_D = abs(rnorm(self.u_D, self.sd_D, T * N))
        e_E = abs(rnorm(self.u_E, self.sd_E, T * N))
        e_G = rnorm(0, self.sd_G, T * N).reshape((T, N))

        obs[0, :self.lag, :] = rnorm(self.init_u_G, self.init_sd_G, self.lag * N).reshape(self.lag, N)
        obs[1, :, :] = (rbin(1, self.p_D, T * N) * e_D).reshape((T, N))
        obs[2, :, :] = (rbin(1, self.p_E, T * N) * e_E).reshape((T, N))
        
        actions = np.random.choice(range(len(self.p_A)), size = T * N, p = self.p_A).reshape((T, N))
            
        ######### Transition
        for t in range(self.lag - 1, self.T_burnin + 5):
            if self.behav is not None:
                S = self.conc_SA_2_state(obs, actions, t, multiple_N = True).T
                actions[t, :] = np.squeeze(self.behav.sample_A(S))
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t + 1)
            obs[0, t + 1, :] = self.step(states = states, errors = np.array([e_G[t, :]]))
        actions = actions.astype(float)

        return obs[:, self.T_burnin:, :], e_G[self.T_burnin:, :], actions[self.T_burnin:, :] 
    ##################################################################################################################################################################
    def conc_single(self, obs, actions, t):
        # obs = [3, T]
        # actions = [T]
        As = actions[(t - self.lag + 1):t]
        S = obs[:, (t - self.lag + 1):(t + 1)]
        s = S.ravel(order='C')
        s = np.append(s, As)
        return s
############################################################################################################################################

    def step(self, states = None, errors = None):
        return np.array(self.CONST).reshape((1, 1)) + self.tran_mat.dot(states) + errors
    
    def simu_one_seed(self, seed = 42, N = None, T = None):
        """ Simulate N patient trajectories with length T, calibrated from the Ohio dataset.
        Returns:
            trajs = [traj], where traj [[S, A, R, SS]]
            
        """
        if N is None:
            N = self.N
        if T is None:
            T = self.T
        np.random.seed(seed)
        # Initialization
        obs, e_G, actions = self.init_MDPs(seed = seed, N = N)
            
        for t in range(self.lag - 1, T - 1):
            if self.behav is not None:
                S = self.conc_SA_2_state(obs, actions, t, multiple_N = True).T
                actions[t, :] = np.squeeze(self.behav.sample_A(S))
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t + 1)
            obs[0, t + 1, :] = self.step(states = states, errors = np.array([e_G[t, :]]))
        actions = actions.astype(float)
        
        ######### Collection: obs = (3, T, N), actions = (T, N)
        # conc_SA_2_state(obs, actions, t, multiple_N = False, J = 4) 
        trajs = [[] for i in range(N)]
        for t in range(self.lag - 1, T - 1):  # -1, 10/01/2020
            conc_s = self.conc_SA_2_state(obs, actions, t, multiple_N = True) # (dim_with_J, N)
            conc_ss = self.conc_SA_2_state(obs, actions, t + 1, multiple_N = True)# (dim_with_J, N)
            As = actions[t, :]
            Rs = self.Glucose2Reward(conc_ss[-3, :])
            for i in range(N):
                trajs[i].append([conc_s[:, i], As[i], Rs[i], conc_ss[:, i]])
        return trajs
    
    def simu_init_S(self, seed = 42, N = None):
        if N is None:
            N = self.N
        trajs = self.simu_one_seed(seed = seed, N = N, T = 6)

        return arr([trajs[i][0][0] for i in range(N)])
    
############################################################################################################################################
############################################################################################################################################
    def eval_policy(self, pi = None, N = None
                        , gamma = 1, seed = 42, return_init = False, return_value = True, return_init_value = False
                       , init_S = None, init_A = None
                       ):
        """ Evaluate the value of a policy in simulation.
        sample the first four time points，and then begin to follow the policy and collect rewards
        transform into matrix so that linear transition is easier
        Concatenate data for training and evaluation 

        """
        pi.seed = seed
        np.random.seed(seed)
        T = self.T
        if N is None:
            N = self.N
        ##############################
        ## after burn-in
        obs, e_G, actions_init = self.init_MDPs(seed = seed, N = N) # obs = [3, T, N]
        self.e_G = e_G
        actions = np.zeros((T, N)) # store previous actions+
        actions[:(self.lag - 1), :] = actions_init[:(self.lag - 1), :] 
        if init_S is not None:
            obs[:, :self.lag, :] = init_S
            actions[:(self.lag - 1), :] = init_A
        # self.init = [obs.copy(), e_G.copy(), actions[:self.lag, :].copy()]
        ##############################
        curr_time = now()
        if return_init:
            # can be used for references?
            return obs[:, :self.lag, :], actions[:(self.lag - 1), :]

        for t in range(self.lag - 1, T - 1): 
            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, actions, t, multiple_N = True).T #  N * dim
            actions[t, :] = np.squeeze(pi.sample_A(S)) # s [N * dx] -> actions [N * 1] -> 1 * N

            # next observations: based on ..., t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t + 1)
            obs[0, t + 1, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

        ##############################
        if return_value:
            discounted_values, average_values = self.cal_reward(obs, gamma)
            if return_init_value:
                return discounted_values
            printR("True Value: mean = {:.2f} with std = {:.2f}".format(np.mean(discounted_values), np.std(discounted_values) / np.sqrt(N)))          
            return mean(discounted_values) # len-N
        else:
            ######### Collection: obs = (3, T, N), actions = (T, N)
            trajs = [[] for i in range(N)]
            for t in range(self.lag - 1, T - 1): 
                conc_s = self.conc_SA_2_state(obs, actions, t, multiple_N = True) # (dim_with_J, N)
                conc_ss = self.conc_SA_2_state(obs, actions, t + 1, multiple_N = True)# (dim_with_J, N)
                As = actions[t, :]
                Rs = self.Glucose2Reward(conc_ss[-3, :])
                for i in range(N):
                    trajs[i].append([conc_s[:, i], As[i], Rs[i], conc_ss[:, i]])
            return trajs

    
    def cal_reward(self, obs, gamma):
        all_rewards = self.Glucose2Reward(obs)
        all_rewards = all_rewards[0]
        all_rewards = all_rewards[self.lag:]
        #all_rewards = np.roll(all_rewards[self.lag:], shift = -1, axis = 0)
        all_rewards = np.squeeze(all_rewards)
        #gammas = np.expand_dims(arr([gamma ** j for j in range(all_rewards.shape[0])]), 0)
        discounted_values = sum(r * gamma ** j for j, r in enumerate(all_rewards))
        #discounted_values = np.dot(gammas, all_rewards)
        average_values = np.mean(all_rewards)  # 0
        return discounted_values, average_values

############################################################################################################################################
############################################################################################################################################

    def concatenate_useful_obs(self, obs, actions, t):
        # (dim_with_J, N)
        r = np.vstack([
            obs[0, (t - self.lag):t, :], obs[1, (t - self.lag):t, :],
            obs[2, (t - self.lag):t, :], actions[(t - self.lag):t, :]])
        return r


    def conc_SA_2_state(self, obs, actions, t, multiple_N = False):
        """ to form a lag-J states from history obs and A
        """
        # dim = (3, T, N)
        N = obs.shape[2]
        dim_obs = 3
        s = np.vstack(([
            obs[:, (t - self.lag + 1 ):t, :],
            actions[(t - self.lag + 1):t, :].reshape((1, self.lag - 1, N))])) # extend_dim for first one
        s = s.reshape(((dim_obs + 1) * (self.lag - 1), N), order = 'F')
        obs_0 = obs[:, t, :] # 3 * N
        s = np.vstack([s, obs_0])
        return s
############################################################################################################################################
############################################################################################################################################

    def simu_multi_seeds(self, M, parallel = True):
        if parallel:
            return parmap(self.simu_one_seed, range(M))
        else:
            return [self.simu_one_seed(i) for i in range(M)]

    def eval_trajs(self, trajs = None, gamma = .8):        
        def get_rew(i): 
            glucoses = arr([item[0][-3] for item in trajs[i]])
            rewards = np.roll(self.Glucose2Reward(glucoses), shift = -1).reshape(-1, 1)
            discounted_value = sum(r * gamma ** t for t, r in enumerate(rewards))
            discounted_value = round(discounted_value, 2)
            average_value = round(np.mean(rewards))
            return [rewards, discounted_value, average_value]
        res = parmap(get_rew, range(len(trajs)))
        discounted_values = [a[1] for a in res]
        average_values = [a[2] for a in res]
        return discounted_values, average_values

############################################################################################################################################
############################################################################################################################################
    def reset(self, T = 1000):
        self.seed += 1
        if self.seed > 1e4:
            self.seed = 42
        self.T = T
        np.random.seed(self.seed)
        ### Initialization
        self.obs = np.zeros((3, T)) 
        self.e_D = abs(rnorm(self.u_D, self.sd_D, T))
        self.e_E = abs(rnorm(self.u_E, self.sd_E, T))
        self.e_G = rnorm(0, self.sd_G, T)
        self.obs[0, :self.lag] = rnorm(self.init_u_G, self.init_sd_G, self.lag)
        self.obs[1, :] = rbin(1, self.p_D, T) * self.e_D
        self.obs[2, :] = rbin(1, self.p_E, T) * self.e_E
        self.t = 3
        self.actions = zeros(T)
        self.actions[:3] = np.zeros(3)
        states = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        return states
    
    def online_step(self, action):
        self.actions[self.t] = action
        # 1. transition
        states = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        SA = np.append(states, action)
        self.t += 1
        next_G = self.CONST + self.tran_mat.dot(SA) + self.e_G[self.t]
        # 2. store
        self.obs[0, self.t] = next_G
        # 3. return 
        observation_ = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        done = (self.t == (self.T - 1))
        reward = self.Glucose2Reward(next_G)
        reward += (randn(1) / 1e4)
        reward = np.squeeze(reward)

        return observation_, reward, done
############################################################################################################################################

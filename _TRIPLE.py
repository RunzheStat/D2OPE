from _util import *
import _Ohio_Simulator as Ohio
import _RL.FQE as FQE_module
import _RL.FQI as FQI
reload(Ohio)
reload(FQE_module)
reload(FQI)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    from _density import omega_SA, omega_SASA
    import _RL.sampler as sampler
    reload(omega_SA)
    reload(omega_SASA)
    import tensorflow as tf

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.backend.set_floatx('float64')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

################################################################################################################################################################################################################################################################################################################################################################################################################################################

class ARE():
    """ ADAPTIVE, EFFICIENT AND ROBUST OFF-POLICY EVALUATION
    for a dataset and a given policy
    , estimate the components (Q, omega, omega_star)
    , and construct the doubly, triply, ... robust estimators for the itegrated value
    """
    def __init__(self, trajs, pi, eval_N = 1000, gpu_number = 0, verbose = 0
                 , L = 2, incomplete_ratio = 20, sepe_A = 0, A_range = [0, 1, 2, 3, 4]
                 , gamma = .9):
        self.trajs = trajs # data: T transition tuple for N trajectories
        self.S, self.A, self.R, self.SS = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        self.N, self.T, self.L = len(trajs), len(trajs[0]), L
        self.S_dims = len(np.atleast_1d(trajs[0][0][0]))
        self.A_range = arr(A_range).astype(np.float64) #set(self.A)
        self.num_A = len(self.A_range)
        self.gamma = gamma
        self.pi = pi
        self.gpu_number = gpu_number
        self.alphas = alphas = [0.05, 0.1]
        self.z_stats = [sp.stats.norm.ppf((1 - alpha / 2)) for alpha in self.alphas]
        self.split_ind = sample_split(self.L, self.N) # split_ind[k] = {"train_ind" : i, "test_ind" : j}
        self.eval_N = eval_N
        self.sepe_A = sepe_A
        self.verbose = verbose
        self.incomplete_ratio = incomplete_ratio
        
        self.value_funcs = []
        self.omegas = []
        self.omegas_values = []
        self.omegas_star = []
        self.omegas_star_values = []
        self.Q_values = {}
        self.DR = {}
        self.raw_Qs = zeros(self.L)
        self.psi_it = []
        self.IS_it = []
        self.psi2_it = []

    ############################################################################################################################
    ###########################################  The three main components #####################################################
    ############################################################################################################################
    def est_Q(self, verbose = 1, test_freq = 10, **FQE_paras):
        """ Q_func(self, S, A = None)
        self.ohio_eval.init_state
        """        
        #########
        for k in range(self.L):
            curr_time = now()
            ##################
            value_func = FQE_module.FQE(policy = self.pi # policy to be evaluated
                     , num_actions = self.num_A, gamma = self.gamma, init_states = self.init_S # used for evaluation
                     , gpu_number = self.gpu_number, **FQE_paras)
            value_func.train([self.trajs[i] for i in self.split_ind[k]["train_ind"]], verbose = verbose, test_freq = test_freq)
            ###########################
            init_V = value_func.init_state_value(init_states = self.init_S)
            ### the behav value is significant affected by the initial
            ## stationary?
            train_traj = [self.trajs[i] for i in self.split_ind[k]["train_ind"]]
            disc_w_init = mean([sum([SA[2] * self.gamma ** t for t, SA in enumerate(traj)]) for traj in train_traj])
            S, A, R, SS = [np.array([item[i] for traj in train_traj for item in traj]) for i in range(4)]
            disc_w_all = np.mean(R / (1 - self.gamma))
            self.raw_Qs[k] = np.mean(init_V)
            self.value_funcs.append(value_func)
            
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            Q_S = self.value_funcs[k].Q_func(S, A)
            Q_SS = self.value_funcs[k].Q_func(SS, self.pi.get_A(SS))
            sampled_Qs = self.value_funcs[k].Q_func(self.init_S, self.pi.get_A(self.init_S))
            self.Q_values[k] = {"Q_S" : Q_S.copy(), "Q_SS" : Q_SS.copy(), "sampled_Qs" : sampled_Qs.copy()}
            if self.verbose:
                printR("behav value: disc_w_init = {:.2f} and disc_w_all = {:.2f}".format(disc_w_init, disc_w_all))
                printR("OPE init_Q: mean = {:.2f} and std = {:.2f}".format(np.mean(init_V)
                                                                                     , np.std(init_V) / np.sqrt(len(self.init_S))))
                printG("<------------- FQE for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
        #########
        self.raw_Q = mean(self.raw_Qs)
    
    def load_Q(self, Q_values = None):
        self.raw_Qs = Q_values["raw_Qs"]
        self.Q_values = Q_values["Q_values"]
        self.raw_Q = mean(self.raw_Qs)
    
    def est_w(self, h_dims = 32, max_iter = 100, batch_size = 32, lr = 0.0002, print_freq = 20, tolerance = 5, rep_loss = 3): 
        for k in range(self.L):
            curr_time = now()
            ###
            omega_func = omega_SA.VisitationRatioModel_init_SA(replay_buffer = sampler.SimpleReplayBuffer(trajs = [self.trajs[i] for i in self.split_ind[k]["train_ind"]])
                            , target_policy = self.pi, A_range = self.A_range, h_dims = h_dims
                            , lr = lr, gpu_number = self.gpu_number, sepe_A = self.sepe_A)

            omega_func.fit(batch_size = batch_size, gamma = self.gamma, max_iter = max_iter
                          , print_freq = print_freq, tolerance= tolerance, rep_loss = rep_loss)
            ### 
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            #########
            omega = omega_func.model.predict_4_VE(inputs = tf.concat([S, A[:,np.newaxis]], axis=-1)) #  (NT,)
            omega = np.squeeze(omega)
            if self.verbose:
                printG("<------------- omega estimation for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
            self.omegas_values.append(omega)
            self.omegas.append(omega_func.model)

    def est_IS(self):
        # use to construct G() and so the original value estimator
        #########################
        for k in range(self.L):
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            ################################
            omega = self.omegas_values[k]
            ISs = (omega * R) / (1 - self.gamma) / np.mean(omega)
            if self.verbose:
                printR("IS for fold {} = {:.2f}".format(k, np.mean(ISs)))
            self.IS_it.append(ISs)
        self.IS_V = self.cal_metric(self.IS_it)

    def est_cond_w(self, h_dims = 32, max_iter = 100, batch_size = 32, print_freq = 20, lr = 0.0002, tolerance= 5, rep_loss = 3): 
        for k in range(self.L):
            curr_time = now()
            ### fit
            omega_func = omega_SASA.VisitationRatioModel_init_SASA(replay_buffer = sampler.SimpleReplayBuffer(trajs = [self.trajs[i] for i in self.split_ind[k]["train_ind"]])
                                    , target_policy = self.pi, A_range = self.A_range, h_dims = h_dims, gpu_number = self.gpu_number
                                                                    , lr = lr, sepe_A = self.sepe_A)
            omega_func.fit(batch_size = batch_size, gamma = self.gamma, max_iter = max_iter, print_freq = print_freq
                          , tolerance= tolerance, rep_loss = rep_loss)
            if self.verbose:
                printG("<------------- omega* estimation for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
                
            self.omegas_star.append(omega_func.model)
            
    ############################################################################################################################
    ###########################################  Value Estimators #####################################################
    ############################################################################################################################
    # store the previous one and do average in the next to debias the previous one.
    def est_double_robust(self):
        # use to construct G() and so the original value estimator
        #########################
        for k in range(self.L):
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            ################################
            omega = self.omegas_values[k]
            Q_S = self.Q_values[k]["Q_S"]
            # the below one is w.r.t deterministic policy pi(); can be updated with the weighted version 
            Q_SS = self.Q_values[k]["Q_SS"]
            bellman_error = R + self.gamma * Q_SS - Q_S
            Q_debias = np.squeeze(omega) * bellman_error / (1 - self.gamma)
            ######### integrated_Q #########
            # how to use?
            sampled_Qs = self.Q_values[k]["sampled_Qs"]
            integrated_Q = np.mean(sampled_Qs)
            if self.verbose:
                printR("integrated_Q for fold {} = {:.2f}".format(k, integrated_Q))
            ######### Putting together #########
            self.psi_it.append(Q_debias + integrated_Q)
            ######### used for TR #########
            self.DR[k] = {"Q_S" : self.Q_values[k]["Q_S"], "Q_SS" : self.Q_values[k]["Q_SS"]
                          , "sampled_Qs" : self.Q_values[k]["sampled_Qs"], "bellman_error" : bellman_error # (NT, )
                          }
        ##############  cal_metric  ##############
        self.DR_V = self.cal_metric(self.psi_it)         
        if self.verbose:
            for k in range(self.L):
                print("DR for fold {} = {:.3f}".format(k, mean(self.psi_it[k])))
            printR("DR: est = {:.2f}, sigma = {:.2f}".format(self.DR_V["V"], self.DR_V["sigma"]))
    
    
    def est_triply_robust(self):
        self.cond_X_on_S, self.cond_X_on_SS, self.cond_X_on_init_S = {}, {}, {}
        self.cond_X_on_S_idx, self.cond_X_on_SS_idx, self.cond_X_on_init_S_idx = {}, {}, {}
        self.Q2_S_debiased, self.Q2_SS_debiased, self.Q2_S_init_S_debiased = {}, {}, {}
        self.large = {"Q2_S_debiased_bef" : {}
                     , "Q2_SS_debiased_bef" : {}
                     , "Q2_init_S_debiased_bef" : {}}
        self.size = len(self.S) // self.incomplete_ratio // self.L
        #########################
        for k in range(self.L):
            curr_time = now()
            # debias for each Q
            ############### re-define the three Q-related terms ###############
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            n = len(S)
            X = np.concatenate([S, A[:,np.newaxis]], axis=-1)
            ############### omega: w(S', A'; S) ###############
            def get_cond_w(main_SA):
                m = len(main_SA)
                np.random.seed(42)
                
                # used_X_idx = [np.random.choice(n, size = size) + i * n for i in range(m)]
                used_X_idx = [np.random.choice(n, size = self.size) for i in range(m)]
                used_X_idx_conc = np.concatenate([used_X_idx[i] + i * n  for i in range(m)])

                X_t = np.vstack([X[used_X_idx[i]] for i in range(m)])
                SASA = np.hstack([X_t, np.repeat(main_SA, self.size, axis=0)])
                
                omega_star = self.omegas_star[k].predict_4_VE(SASA) # n2
                omega_star = np.squeeze(omega_star)
                omega_star = omega_star.reshape(m, self.size, order = "C") # "F"
                omega_star = omega_star / np.mean(omega_star, 1, keepdims = True)
                return omega_star, used_X_idx
            self.cond_X_on_S[k], self.cond_X_on_S_idx[k] = get_cond_w(X)

            SSA = np.concatenate([SS, self.pi.get_A(SS)[:,np.newaxis]], axis=-1)
            self.cond_X_on_SS[k], self.cond_X_on_SS_idx[k] = get_cond_w(SSA)

            init_S_A = np.concatenate([self.init_S, self.pi.get_A(self.init_S)[:,np.newaxis]], axis=-1)
            self.cond_X_on_init_S[k], self.cond_X_on_init_S_idx[k] = get_cond_w(init_S_A)

            # printG("<------------- TR density estimation for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
            def get_debiased_Q(Q_before_debias, omega_star, idx = None):
                """ 
                input: Q functions and conditional density
                output: debiased Q function

                m: len of S, to be debiased
                n: sample size, Q_debias, the bellman error term
                """
                m = len(omega_star)
                ##### Q^(2)(S, A)
                # repeat into n2, multiply with omega, and calculate the average
                Q2_S_debias = np.repeat(self.DR[k]["bellman_error"], m, axis=0).reshape(m, n, order = "F") / (1 - self.gamma) # n2. tile is for t = 0, repeat is for do average
                Q2_S_debias = arr([Q2_S_debias[i, idx[i]] for i in range(len(Q2_S_debias))])
                Q2_S_debias_bef = omega_star * Q2_S_debias # n2. tile is for t = 0, repeat is for do average
                # do average
                Q2_S_debias = np.mean(Q2_S_debias_bef, 1) # n
                # debiased
                Q2_S_debiased = Q_before_debias + Q2_S_debias
                return Q2_S_debiased, Q_before_debias[:, np.newaxis] + Q2_S_debias_bef

            self.Q2_S_debiased[k], self.large["Q2_S_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["Q_S"], omega_star = self.cond_X_on_S[k], idx = self.cond_X_on_S_idx[k]) # (NT, )
            self.Q2_SS_debiased[k], self.large["Q2_SS_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["Q_SS"], omega_star = self.cond_X_on_SS[k], idx = self.cond_X_on_SS_idx[k]) # (NT, )
            self.Q2_S_init_S_debiased[k], self.large["Q2_init_S_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["sampled_Qs"], omega_star = self.cond_X_on_init_S[k], idx = self.cond_X_on_init_S_idx[k])

            ## construct the DR with debiased Qs
            omega = self.omegas_values[k]
            omega /= np.mean(omega)
            self.Q_debias = omega * (R + self.gamma * self.Q2_SS_debiased[k] - self.Q2_S_debiased[k]) 
            self.Q_debias /= (1 - self.gamma)
            self.integrated_Q = np.mean(self.Q2_S_init_S_debiased[k])
            ######### Putting together #########

            self.psi2_it.append(self.Q_debias + self.integrated_Q)
            if self.verbose:
                printG("<------------- TR for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
        #########################        
        self.TR_V = self.cal_metric(self.psi2_it)
        if self.verbose:
            for k in range(self.L):
                print("TR for fold {} = {:.3f}, with integrated_Q = {:.3f}".format(k, mean(self.psi2_it[k])
                                                                              , np.mean(self.Q2_S_init_S_debiased[k])))


    ####################################################################################################
    
    def est_quad_robust(self):
        # quadruple; infinite
        self.Q3_S_debiased, self.Q3_SS_debiased, self.Q3_S_init_S_debiased = {}, {}, {}
        self.psi3_it = []
        self.integrated_Q2 = {}
        for k in range(self.L):
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            n = len(S)
            X = np.concatenate([S, A[:,np.newaxis]], axis=-1)
            Q2_bellman = np.repeat(np.expand_dims(R, 1), self.size, axis = 1) + self.gamma * self.large["Q2_SS_debiased_bef"][k] - self.large["Q2_S_debiased_bef"][k] # [i', i] = [X, X] = [NT, NT]
            """
            bef: [S, X], [0, i]
            bellman: [X, X], [i', i]
            omega: [S, X], [0, i']
            duplication exists here
            """        

            def debias(bef, idx, omega_star):
                return np.mean(bef + np.vstack([a[np.newaxis, :].dot(Q2_bellman[idx[i]]) for i, a in enumerate(omega_star)]) / n / (1 - self.gamma), 1)

            self.Q3_S_debiased[k] = debias(bef = self.large["Q2_S_debiased_bef"][k], idx = self.cond_X_on_S_idx[k], omega_star = self.cond_X_on_S[k])
            self.Q3_SS_debiased[k] = debias(bef = self.large["Q2_SS_debiased_bef"][k], idx = self.cond_X_on_SS_idx[k], omega_star = self.cond_X_on_SS[k])

            self.integrated_Q2[k] = debias(bef = self.large["Q2_init_S_debiased_bef"][k], idx = self.cond_X_on_init_S_idx[k], omega_star = self.cond_X_on_init_S[k])
            self.integrated_Q2[k] = np.mean(self.integrated_Q2[k])
            omega = self.omegas_values[k]
            omega /= np.mean(omega)
            self.Q2_debias = omega * (R + self.gamma * self.Q3_SS_debiased[k] - self.Q3_S_debiased[k]) / (1 - self.gamma)

            ######### Putting together #########

            self.psi3_it.append(self.Q2_debias + self.integrated_Q2[k])

        self.QR_V = self.cal_metric(self.psi3_it)
        if self.verbose:
            for k in range(self.L):
                print("QR for fold {} = {:.3f}, with integrated_Q = {:.3f}".format(k, mean(self.psi3_it[k])
                                                                              , self.integrated_Q2[k]))

    def is_diff(self, old, new):
        old = np.concatenate(old)
        new = np.concatenate(new)
        diff = np.mean(new) - np.mean(old)
        std = np.std(arr(new) - arr(old)) / np.sqrt(len(old))
        z = np.abs(diff) / std
        alpha = 0.05
        if z > sp.stats.norm.ppf((1 - alpha / 2)):
            return True
        else:
            return False
    ############################################################################################################################
    ###########################################  Evaluation #####################################################
    ############################################################################################################################
    def cal_metric(self, psi_it):
        """ psi_it is a list (N, T) of value estimates """
        psi_it = np.concatenate(psi_it)
        
        V = np.mean(psi_it)
        sigma = np.std(psi_it) #sqrt(np.sum((psi_it - V) ** 2) / (self.N * self.T - 1))
        CIs = [[V - z_stat * (sigma / sqrt(len(psi_it))) , V + z_stat * (sigma / sqrt(len(psi_it)))]
                                for z_stat in self.z_stats]
        metrics = {"V" : V, "sigma" : sigma / sqrt(len(psi_it))
                     , "CIs" : np.array(CIs)}
        return metrics.copy()

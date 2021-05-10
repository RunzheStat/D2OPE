from _util import *
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class ARE_discrete():
    """ ADAPTIVE, EFFICIENT AND ROBUST OFF-POLICY EVALUATION
    for a dataset and a given policy
    , estimate the components (Q, omega, omega_star)
    , and construct the doubly, triply, ... robust estimators for the itegrated value
    """
    def __init__(self, trajs, pi, eval_N = 1000, gpu_number = 0
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
    
        self.incomplete_ratio = incomplete_ratio
        
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        
        self.value_funcs = []
        self.omegas = []
        self.omegas_values = []
        self.omegas_star = []
        self.omegas_star_values = []
        self.Q_values = {}
        self.DR = {}
        self.raw_Qs = zeros(self.L)
        self.psi_it = []
        self.psi2_it = []

    def load_true(self, true_Q, true_omega, true_omega_star):
        for k in range(self.L):
            init_V = true_Q.init_state_value(init_states = self.init_S)

            self.raw_Qs[k] = np.mean(init_V)
            self.value_funcs.append(true_Q)    
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            Q_S = self.value_funcs[k].Q_func(S, A)
            Q_SS = self.value_funcs[k].Q_func(SS, self.pi.get_A(SS))
            sampled_Qs = self.value_funcs[k].Q_func(self.init_S, self.pi.get_A(self.init_S))
            self.Q_values[k] = {"Q_S" : Q_S.copy(), "Q_SS" : Q_SS.copy(), "sampled_Qs" : sampled_Qs.copy()}
            omega = true_omega.get_omega(S, A) #  (NT,)
            omega = np.squeeze(omega)
            self.omegas_values.append(omega)
            self.omegas.append(true_omega)
            self.omegas_star.append(true_omega_star)
        self.raw_Q = np.mean(self.raw_Qs)
            
    ############################################################################################################################
    ###########################################  Value Estimators #####################################################
    ############################################################################################################################
    # store the previous one and do average in the next to debias the previous one.
    def est_double_robust(self, re_scale = False):
        # use to construct G() and so the original value estimator
        #########################
        for k in range(self.L):
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            ################################
            omega = self.omegas_values[k]
            if re_scale:
                omega /= np.mean(omega)
            Q_S = self.Q_values[k]["Q_S"]
            # the below one is w.r.t deterministic policy pi(); can be updated with the weighted version 
            Q_SS = self.Q_values[k]["Q_SS"]
            bellman_error = R + self.gamma * Q_SS - Q_S
            Q_debias = np.squeeze(omega) * bellman_error / (1 - self.gamma)
            ######### integrated_Q #########
            # how to use?
            sampled_Qs = self.Q_values[k]["sampled_Qs"]
            integrated_Q = np.mean(sampled_Qs)
            MWLs = (omega * R) / (1 - self.gamma) / np.mean(omega)
            # printR("MWLs for fold {} = {:.2f}".format(k, np.mean(MWLs)))
            ######### Putting together #########
            self.psi_it.append(Q_debias + integrated_Q)
            ######### used for TR #########
            self.DR[k] = {"Q_S" : self.Q_values[k]["Q_S"], "Q_SS" : self.Q_values[k]["Q_SS"]
                          , "sampled_Qs" : self.Q_values[k]["sampled_Qs"], "bellman_error" : bellman_error # (NT, )
                          , "MWLs" : MWLs
                          }
        ##############  cal_metric  ##############
        self.DR_V = self.cal_metric(self.psi_it)         
        
        
    def est_triply_robust(self):
        # can be combined with the last?
        self.cond_X_on_S, self.cond_X_on_SS, self.cond_X_on_init_S = {}, {}, {}
        self.cond_X_on_S_idx, self.cond_X_on_SS_idx, self.cond_X_on_init_S_idx = {}, {}, {}
        self.Q2_S_debiased, self.Q2_SS_debiased, self.Q2_S_init_S_debiased = {}, {}, {}
        self.large = {"Q2_S_debiased_bef" : {}
                     , "Q2_SS_debiased_bef" : {}
                     , "Q2_init_S_debiased_bef" : {}}
        self.cond_w = {}
        #########################
        for k in range(self.L):
            curr_time = now()
            # debias for each Q
            ############### re-define the three Q-related terms ###############
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            n = len(S)
            X = np.concatenate([S[:,np.newaxis], A[:,np.newaxis]], axis=-1)
            
            ## reduce unique number of idx set to save time cost             
            x_uniq_set, X_idx = np.unique(X, return_inverse=True, axis = 0)
            
            ############### omega: w(S', A'; S, A) ###############
            def get_cond_w(main_S):
                m = len(main_S)
                X_t = np.vstack([X for i in range(m)])
                size = n
                
                SAS = np.hstack([X_t, np.repeat(main_S, size, axis=0)])
                main_S = np.repeat(main_S, size, axis=0)
                omega_star = self.omegas_star[k].get_omega_star(X_t[:, 0], X_t[:, 1], main_S[:, 0], main_S[:, 1]) # n2
                omega_star = np.squeeze(omega_star)
                omega_star = omega_star.reshape(m, size, order = "C") # "F"
                omega_star = omega_star / np.mean(omega_star, 1, keepdims = True)
                return omega_star
            ############################################################
            self.cond_w[k] = get_cond_w(x_uniq_set)
            
            def get_row_idx(mat):
                idxs = []
                for x in x_uniq_set:
                    idxs.append(np.where((mat == x).all(axis=1))[0])
                return idxs
            
            self.cond_X_on_S_idx[k] = get_row_idx(X)
            
            SSA = np.concatenate([SS[:,np.newaxis], self.pi.get_A(SS)[:,np.newaxis]], axis=-1)
            self.cond_X_on_SS_idx[k] = get_row_idx(SSA)
            
            init_S_A = np.concatenate([self.init_S[:,np.newaxis], self.pi.get_A(self.init_S)[:,np.newaxis]], axis=-1)
            self.cond_X_on_init_S_idx[k] = get_row_idx(init_S_A)
            
#             self.cond_X_on_S[k], self.cond_X_on_S_idx[k] = get_cond_w(X)
#             self.cond_X_on_SS[k], self.cond_X_on_SS_idx[k] = get_cond_w(SSA)
#             self.cond_X_on_init_S[k], self.cond_X_on_init_S_idx[k] = get_cond_w(init_S_A)

            # all time cost here
            def get_debiased_Q(Q_before_debias, idx = None):
                """ 
                input: Q functions and conditional density
                output: debiased Q function

                m: len of S, to be debiased
                n: sample size, Q_debias, the bellman error term
                """
                m = len(Q_before_debias)
                ##### Q^(2)(S, A)
                # repeat into n2, multiply with omega, and calculate the average
                Q2_S_debias = np.repeat(self.DR[k]["bellman_error"], m, axis=0).reshape(m, n, order = "F") / (1 - self.gamma) # n2. tile is for t = 0, repeat is for do average
                
                omega_star = zeros((m, n))
                for i, cond_w in zip(idx, self.cond_w[k]):
                    omega_star[i] = cond_w
                Q2_S_debias_bef = omega_star * Q2_S_debias # n2. tile is for t = 0, repeat is for do average
                # do average
                Q2_S_debias = np.mean(Q2_S_debias_bef, 1) # n
                # debiased
                Q2_S_debiased = Q_before_debias + Q2_S_debias
                return Q2_S_debiased, Q_before_debias[:, np.newaxis] + Q2_S_debias_bef
            # no A -> no difference between SS and S? very close
            # all three Q-related NEED TO be biased

            self.Q2_S_debiased[k], self.large["Q2_S_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["Q_S"],  idx = self.cond_X_on_S_idx[k]) # (NT, )
            self.Q2_SS_debiased[k], self.large["Q2_SS_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["Q_SS"], idx = self.cond_X_on_SS_idx[k]) # (NT, )
            self.Q2_S_init_S_debiased[k], self.large["Q2_init_S_debiased_bef"][k] = get_debiased_Q(Q_before_debias = self.DR[k]["sampled_Qs"], idx = self.cond_X_on_init_S_idx[k])

            ## construct the DR with debiased Qs
            omega = self.omegas_values[k]
            omega /= np.mean(omega)
            self.Q_debias = omega * (R + self.gamma * self.Q2_SS_debiased[k] - self.Q2_S_debiased[k]) 
            self.Q_debias /= (1 - self.gamma)
            self.integrated_Q = np.mean(self.Q2_S_init_S_debiased[k])
            ######### Putting together #########

            self.psi2_it.append(self.Q_debias + self.integrated_Q)
        #########################        
        self.TR_V = self.cal_metric(self.psi2_it)


    ####################################################################################################
    
    def est_quad_robust(self):
        self.Q3_S_debiased, self.Q3_SS_debiased, self.Q3_S_init_S_debiased = {}, {}, {}
        self.psi3_it = []
        self.integrated_Q2 = {}
        for k in range(self.L):
            S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            n = len(S)
            X = np.concatenate([S[:,np.newaxis], A[:,np.newaxis]], axis=-1)
            Q2_bellman = np.repeat(np.expand_dims(R, 1), n, axis = 1) + self.gamma * self.large["Q2_SS_debiased_bef"][k] - self.large["Q2_S_debiased_bef"][k] # [i', i] = [X, X] = [NT, NT]
            
            uniq_debias = self.cond_w[k].dot(Q2_bellman) / n / (1 - self.gamma)
            uniq_debias_mean = np.mean(uniq_debias, 1)
            

            def debias(bef, idx):
                """
                bef: [0, i], N * size 
                omega_star: [0, i'], N * size
                Q2_bellman: [i', i], N * N -> idx -> size * size [but not necessary all i' used that set of i???]
                """        
                m = len(bef)
                omega_star = zeros(m)
                for i, cond_w in zip(idx, uniq_debias_mean):
                    omega_star[i] = cond_w

                return bef + omega_star


            self.Q3_S_debiased[k] = debias(bef = self.Q2_S_debiased[k], idx = self.cond_X_on_S_idx[k])
            

            self.Q3_SS_debiased[k] = debias(bef = self.Q2_SS_debiased[k], idx = self.cond_X_on_SS_idx[k])

            self.integrated_Q2[k] = debias(bef = self.Q2_S_init_S_debiased[k], idx = self.cond_X_on_init_S_idx[k])
            self.integrated_Q2[k] = np.mean(self.integrated_Q2[k])
            omega = self.omegas_values[k]
            omega /= np.mean(omega)
            self.Q2_debias = omega * (R + self.gamma * self.Q3_SS_debiased[k] - self.Q3_S_debiased[k]) / (1 - self.gamma)

            ######### Putting together #########

            self.psi3_it.append(self.Q2_debias + self.integrated_Q2[k])

        self.QR_V = self.cal_metric(self.psi3_it)

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

####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class simuCricle():
    def __init__(self, gamma, pi):
        self.seed = 42
        self.gamma = gamma
        self.pi = pi
        
    def calculate_rewards(self, states):
        """ state 0 -> reward 1"""
        return (states == 0).astype(np.int) 
    
    def take_random_action(self, states):
        np.random.seed(self.seed)
        self.seed += 1
        As = np.random.choice(a = [-1, 1], size = len(states), p = [1 / 2, 1 / 2])
        return  As
        
    def take_action(self, states, policy):
        if policy == "behav":
            return self.take_random_action(states)
        if policy == "tp":
            return self.pi.get_A(states)
    
    def transit(self, S, A):
        """ now: stochastic environment """
        np.random.seed(self.seed)
        self.seed += 1
        S_if = (S + A) % 3
        success = np.random.uniform(0, 1, len(S)) > 0.1
        return np.select([success, ~success], [S_if, S])
    
    def simu_trajs(self, N, T, policy = "behav", init_SA = None, seed = 42):
        """
        1. init
        2. receive actions
        2. transition
        """

        self.seed = seed
        np.random.seed(self.seed)
        self.seed += 1
        if init_SA is not None:
            init_S = np.repeat(init_SA[0], N)
        else:
            init_S = np.random.choice(a = 3, size = N, p = [1 / 3, 1 / 3, 1 / 3])
        S = init_S.copy()
        
        Ss, As, Rs, SSs = [], [], [], []
        for t in range(T):
            if init_SA is not None and t == 0:
                A = np.repeat(init_SA[1], N)
            else:
                A = self.take_action(S, policy)
            SS = self.transit(S, A)
            R = self.calculate_rewards(SS)
            Ss.append(S.copy())
            As.append(A.copy())
            Rs.append(R.copy())
            SSs.append(SS.copy())
            S = SS.copy()

        trajs = [[] for i in range(N)]
        for i in range(N):
            for t in range(T):
                trajs[i].append([Ss[t][i], As[t][i], Rs[t][i], SSs[t][i]])

        return trajs, [Ss, As, Rs, SSs]
    

    def get_omega(self, details_behav, details_tp):
        T = len(details_behav[0])
        N = len(details_behav[0][0])
        
        cnt_SA_behva = np.zeros((3, 2))
        cnt_SA_tp = np.zeros((3, 2))
        
        for t in range(T):
            for i in range(N):
                ###
                S = details_behav[0][t][i]
                A = details_behav[1][t][i]
                A = np.max([A, 0])
                cnt_SA_behva[S, A] += 1
                ###
                S = details_tp[0][t][i]
                A = details_tp[1][t][i]
                A = np.max([A, 0])
                cnt_SA_tp[S, A] += self.gamma ** t
        cnt_SA_tp *= (1 - self.gamma)
        
        den = cnt_SA_behva / np.sum(cnt_SA_behva)
        num = cnt_SA_tp / np.sum(cnt_SA_tp)
        
        omega = num / den

        return omega

        
    def get_V(self, rews):
        T = len(rews)
        N = len(rews[0])
        Vs = [sum(self.gamma ** t * rews[t][i] for t in range(T)) for i in range(N)]
        return mean(Vs), std(Vs) / sqrt(N)


class toy_pi():
    def __init__(self, p = 0.9):
        self.seed = 42
    def get_A(self, states):
        """ the same with action above"""
        np.random.seed(self.seed)
        self.seed += 1
        N = len(states)
        is_state_0 = (states == 0).astype(np.int) 
        ### if not at 0 and dice = 1, then back to 0
        is_back_to_0 = np.repeat(1, N) 
        state_back_to_0 = ((states - 1.5) * 2).astype(np.int)
        state_no_back_to_0 = -state_back_to_0
        state_is_back_to_0 = state_back_to_0
        
        ### if at 0, then totally random
        state_out_0 = np.random.choice(a = [-1, 1], size = N, p = [0.5, 0.5])
        
        As = is_state_0 * state_out_0 + (1 - is_state_0) * state_is_back_to_0
        As = As.astype(np.int)        
        return As

####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class TrueQ():
    def __init__(self, Q):
        self.Q = Q
        self.A_range = [-1, 1]
        
    def Q_func(self, states, actions):
        As = (actions + 1) // 2
        return self.Q[states, As]
    
    def V_func(self, states):
        Q0 = self.Q_func(states, np.repeat(-1, len(states)))
        Q1 = self.Q_func(states, np.repeat(1, len(states)))
        return np.max(np.hstack([Q0, Q1]), axis = -1)
    
    def init_state_value(self, init_states):
        return self.V_func(init_states)

class TrueOmega():
    def __init__(self, omega):
        self.omega = omega
    def get_omega(self, Ss, actions):
        As = (actions + 1) // 2
        return self.omega[Ss, As]

class TrueOmega_star():
    def __init__(self, omega_star):
        self.omega_star = omega_star
        
    def get_omega_star(self, Ss, actions, init_S, init_A):        
        As = (actions + 1) // 2
        init_A = (init_A + 1) // 2
        return self.omega_star[init_S, init_A, Ss, As]

####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#     def get_omega(self, details_behav, details_tp):
#         T = len(details_behav[0])
#         N = len(details_behav[0][0])
        
#         cnt_SA_behva = np.zeros((3, 2))
#         cnt_SA_tp = np.zeros((3, 2))

#         S = np.concatenate(details_behav[0])
#         A = np.concatenate(details_behav[1])
#         A = np.clip(A, 0, None)
#         A = A.astype(np.int)
        
#         for s, a in zip(S, A):
#             cnt_SA_behva[s, a] += 1

#         S = np.concatenate(details_tp[0])
#         A = np.concatenate(details_tp[1])
#         A = np.clip(A, 0, None)
#         A = A.astype(np.int)
#         tt = np.concatenate([np.repeat(t, N) for t in range(T)])

#         for s, a, t in zip(S, A, tt):
#             cnt_SA_tp[s, a] += self.gamma ** t
#         cnt_SA_tp *= (1 - self.gamma)        
        
#         den = cnt_SA_behva / np.sum(cnt_SA_behva)
#         num = cnt_SA_tp / np.sum(cnt_SA_tp)
        
#         omega = num / den

#         return omega


# class toy_pi():
#     def __init__(self, p = 0.9):
#         self.seed = 42
#         #self.p = p
#     def get_A(self, states):
#         """ the same with action above"""
#         np.random.seed(self.seed)
#         self.seed += 1
#         N = len(states)
#         is_state_0 = (states == 0).astype(np.int) 
#         ### if not at 0 and dice = 1, then back to 0
#         is_back_to_0 = np.repeat(1, N) #np.random.choice(a = 2, size = N, p = [1 - self.p, self.p])
#         is_back_to_0 = is_back_to_0.astype(np.int) 
#         state_back_to_0 = ((states - 1.5) * 2).astype(np.int)
#         state_no_back_to_0 = -state_back_to_0
#         state_is_back_to_0 = is_back_to_0 * state_back_to_0 + (1 - is_back_to_0) * state_no_back_to_0
        
#         ### if at 0, then totally random
#         state_out_0 = np.random.choice(a = [-1, 1], size = N, p = [0.5, 0.5])
        
#         As = is_state_0 * state_out_0 + (1 - is_state_0) * state_is_back_to_0
#         As = As.astype(np.int)        
#         return As

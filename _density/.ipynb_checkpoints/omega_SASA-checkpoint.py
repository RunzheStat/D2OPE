import numpy as np
import time
now = time.time

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.backend.set_floatx('float64')

class VisitationRatioModel_init_SASA():
    """ w(S', A'; S) <- model
    just change (SASA) to (SAS)
    see the derivations on page 2, the supplement of VE
    should be almost the same with SASA
    
    can we just delete? IS? 
    the same estimation?
        n2 = n^2        
        X = [S, A]
        XX = [X, X]
        BS = batch_size        
        S, A | S_0 = S
        S_t is the initial state

    """        

    def __init__(self, replay_buffer
                 , target_policy, A_range
                 , h_dims, sepe_A = 0
                 , A_dims = 1, gpu_number = 0, stochastic_policy = False
                 , lr = 1e-3, w_clipping_val = 0.5, w_clipping_norm = 1.0, beta_1=0.5):
        self.model = Omega_SASA_Model(replay_buffer.S_dims, h_dims, A_dims, gpu_number)
        self.S_dims = replay_buffer.S_dims
        self.A_dims = A_dims
        self.gpu_number = gpu_number
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.A_range = A_range
        self.n_A = len(A_range)
        self.losses = []    
        self.sepe_A = sepe_A
        self.single_median = False
        self.A_factor_over_S = 10
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                lr, decay_steps = 100, decay_rate = 0.99
                                                )
                                                  , clipnorm = w_clipping_norm, clipvalue = w_clipping_val)
        # self.optimizer = tf.keras.optimizers.Adam(lr = lr, beta_1 = beta_1, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
        
    def _compute_medians(self, n= 32, rep = 20):
        # to save memory, we do it iteratively            
        with tf.device('/gpu:' + str(self.gpu_number)):
#             if self.sepe_A:
#                 medians_11_22, medians_12_33, medians_12_34 = [tf.zeros([(self.S_dims + self.A_dims + self.S_dims)]
#                                                                         , tf.float64) for _ in range(3)]
#                 for i in range(rep):
#                     transitions = self.replay_buffer.sample(n)
#                     S, A = transitions[0], transitions[1] 
#                     X = tf.concat([S, A[:,np.newaxis]], axis=-1)
#                     #################
#                     XX = tf.concat([X, S], axis=-1) # n
#                     dxx = tf.repeat(XX, n, axis=0) - tf.tile(XX, [n, 1])
#                     medians_11_22 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5
#                     #################
#                     Xr_Xt = tf.concat([tf.repeat(X, n, axis=0),tf.tile(S, [n, 1])], axis=-1) # n2
#                     dxx = tf.repeat(Xr_Xt, n, axis=0) - tf.tile(XX, [n ** 2, 1]) # n3 x ...
#                     medians_12_33 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p
#                     #################
#                     dxx = tf.repeat(Xr_Xt, n ** 2, axis=0) - tf.tile(Xr_Xt, [n ** 2, 1]) # n4 x ...
#                     medians_12_34 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p

#             else:
            if self.single_median:
                medians_11_22, medians_12_33, medians_12_34 = 0, 0, 0
            else:
                medians_11_22, medians_12_33, medians_12_34 = [tf.zeros([(self.S_dims + self.A_dims + self.S_dims + self.A_dims)], tf.float64) for _ in range(3)]
            for i in range(rep):
                transitions = self.replay_buffer.sample(n)
                S, A = transitions[0], transitions[1] 
                X = tf.concat([S, A[:,np.newaxis]], axis=-1)
                #################
                XX = tf.concat([X, X], axis=-1) # n
                dxx = tf.repeat(XX, n, axis=0) - tf.tile(XX, [n, 1])
                if self.single_median:
                    medians_11_22 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_11_22 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5
                #################
                Xr_Xt = tf.concat([tf.repeat(X, n, axis=0),tf.tile(X, [n, 1])], axis=-1) # n2
                dxx = tf.repeat(Xr_Xt, n, axis=0) - tf.tile(XX, [n ** 2, 1]) # n3 x ...
                if self.single_median:
                    medians_12_33 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_12_33 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p
                #################
                dxx = tf.repeat(Xr_Xt, n ** 2, axis=0) - tf.tile(Xr_Xt, [n ** 2, 1]) # n4 x ...
                if self.single_median:
                    medians_12_34 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_12_34 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p
        dim = self.S_dims * 2 + self.A_dims + self.A_dims
        return medians_11_22 / rep * dim, medians_12_33 / rep * dim, medians_12_34 / rep * dim
    
    """ median, laplacian, etc, are all not correct """
    
    
    def _cal_dist(self, X1 = None, X2 = None, X1_X2 = None, median = None):
        """Laplacian Kernel
        pairwise difference: Delta. exp(-diff)
        """
        X1_X2_dim = (self.S_dims + self.A_dims) * 2
        SA_dim = self.S_dims + self.A_dims
        S_dims = self.S_dims
#         if not self.sepe_A:
        if X1_X2 is not None:
            if self.single_median:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2), axis=-1) / median ) 
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2 / median), axis=-1)) * tf.cast(X1_X2[:, S_dims] == 0, tf.float64)
            dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, SA_dim:]), axis=-1) != 0, tf.float64)
            return dist
        else:
            if self.single_median:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=-1) / median) 
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs((X1 - X2) / median), axis=-1)) * tf.cast(X1[:, S_dims] == X2[:, S_dims], tf.float64)
            dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, SA_dim:] - X2[:, SA_dim:]), axis=-1) != 0, tf.float64)
            return dist
#         else:
#             if X1_X2 is not None:
#                 dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2) / median, axis=-1))  * tf.cast(X1_X2[:, S_dims] == 0, tf.int32)
#                 dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :S_dims]), axis=-1) != 0, tf.float64)
#                 return dist
#             else:
#                 dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2) / median, axis=-1)) * tf.cast(X1[:, S_dims] == X2[:, S_dims], tf.float64)
#                 dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :S_dims] - X2[:, :S_dims]), axis=-1) != 0, tf.float64)
#                 return dist

    
    def repeat(self, X, rep):
        return tf.repeat(X, rep, axis=0)
    def tile(self, X, rep):
        return tf.tile(X, [rep, 1])
    def _compute_loss(self, S1, A1, S2, A2, S11, A11, SS11, S22, A22, SS22
                      , gamma, mean_omega, mean_omega2):
        # re-scale A to give it more importance over S, numerically
        A1 *= self.A_factor_over_S
        A11 *= self.A_factor_over_S
        A2 *= self.A_factor_over_S
        A22 *= self.A_factor_over_S
        with tf.device('/gpu:' + str(self.gpu_number)):
            BS = self.BS
            n = self.BS
            ###########################################################################
            X1 = tf.concat([S1, A1[:,np.newaxis]], axis=-1)
            X11 = tf.concat([S11, A11[:,np.newaxis]], axis=-1)
            X2 = tf.concat([S2, A2[:,np.newaxis]], axis=-1)
            X22 = tf.concat([S22, A22[:,np.newaxis]], axis=-1)

            X1_t = self.tile(X1, BS)
            X2_t = self.tile(X2, BS)
            X1_r = self.repeat(X1, BS) # n2 x ...       
            S1_r = self.repeat(S1, BS) # n2 x ...       
            X2_r = self.repeat(X2, BS) # n2 x ...       
            S2_r = self.repeat(S2, BS) # n2 x ...       

            XX1 = tf.concat([X1, X1], axis=-1) # n
            XX2 = tf.concat([X2, X2], axis=-1) # n
            X11r_X1t = tf.concat([tf.repeat(X11, n, axis=0),tf.tile(X1, [n, 1])], axis=-1) # n2
            X22r_X2t = tf.concat([tf.repeat(X22, n, axis=0),tf.tile(X2, [n, 1])], axis=-1) # n2

            
            ###### probability ######
            # if not stochastic, we can only care about related actions and save lots of time
            A_of_SS11 = tf.convert_to_tensor(self.target_policy.get_A(SS11), dtype=tf.float64) 
            SS11_a = tf.concat([SS11, A_of_SS11[:, np.newaxis]], axis = -1) 
            SS11_r_a = self.repeat(SS11_a, BS)
            
            A_of_SS22 = tf.convert_to_tensor(self.target_policy.get_A(SS22), dtype=tf.float64) 
            SS22_a = tf.concat([SS22, A_of_SS22[:, np.newaxis]], axis = -1) 
            SS22_r_a = self.repeat(SS22_a, BS)

            ###### omega (normalization) ######
            omega_11_1_r_t = self.model.call(X11r_X1t) # n2
            omega_22_2_r_t = self.model.call(X22r_X2t)
                
            omega_11_1_r_t /= (tf.tile(tf.expand_dims(mean_omega, 1), [n,1]))
            omega_11_1_r_t = tf.squeeze(omega_11_1_r_t)

            omega_22_2_r_t /= (tf.tile(tf.expand_dims(mean_omega2, 1), [n,1]))
            omega_22_2_r_t = tf.squeeze(omega_22_2_r_t)

            #################################### part 1, n3 #################################### 
            E_K_1_1 = self._cal_dist(X1 = self.repeat(tf.concat([SS11_r_a, X1_t], axis = -1), BS)
                                          , X2 = self.tile(XX2, BS ** 2)
                                          , median = self.medians_n3)# (n3), rr_rt - tt_tt
            K_r_t_1_1 = self._cal_dist(X1 = self.repeat(X11r_X1t, BS), X2 = self.tile(XX2, BS ** 2), median = self.medians_n3)
            part1_1_bef_sum = tf.squeeze(self.repeat(omega_11_1_r_t, BS)) * (gamma * E_K_1_1 - K_r_t_1_1) * self.del_dup_n3
            part1_1 = (1 - gamma) * tf.reduce_mean(part1_1_bef_sum) 

            E_K_2_2 = self._cal_dist(X1 = self.repeat(tf.concat([SS22_r_a, X2_t], axis = -1), BS)
                                          , X2 = self.tile(XX1, BS ** 2)
                                          , median = self.medians_n3)# (n3), rr_rt - tt_tt
            K_r_t_2_2 = self._cal_dist(X1 = self.repeat(X22r_X2t, BS), X2 = self.tile(XX1, BS ** 2), median = self.medians_n3)
            part1_2_bef_sum = tf.squeeze(self.repeat(omega_22_2_r_t, BS)) * (gamma * E_K_2_2 - K_r_t_2_2) * self.del_dup_n3
            part1_2 = (1 - gamma) * tf.reduce_mean(part1_2_bef_sum) 

            part1 = part1_1 + part1_2
            #################################### part 2, n4 #################################### 
            omega_r_t_22_2 = tf.squeeze(self.repeat(omega_22_2_r_t, BS ** 2)) * tf.squeeze(self.tile(tf.expand_dims(omega_11_1_r_t, 1), BS ** 2)) #* self.del_dup_n4 # n4
            omega_t_r_22_2 = tf.squeeze(self.repeat(omega_11_1_r_t, BS ** 2)) * tf.squeeze(self.tile(tf.expand_dims(omega_22_2_r_t, 1), BS ** 2)) #* self.del_dup_n4 # n4

            ######
            K_term = self._cal_dist(X1 = tf.repeat(tf.concat([SS22_r_a, X2_t], axis = -1), BS ** 2, axis=0) 
                                  , X2 = self.tile(tf.concat([SS11_r_a, X1_t], axis = -1), BS ** 2)
                                  , median = self.medians_n4)
            part2_1 = gamma ** 2 * tf.reduce_mean(K_term * omega_r_t_22_2)
            ######
            E_K_part2_1 = self._cal_dist(X1 = self.repeat(tf.concat([SS11_r_a, X1_t], axis = -1), BS ** 2)
                                         , X2 = self.tile(X22r_X2t, BS**2) 
                                         , median = self.medians_n4) # (n4)
            part2_2_1 = tf.reduce_mean(E_K_part2_1 * omega_t_r_22_2)  * gamma 
            
            E_K_part2_2 = self._cal_dist(X1 = self.repeat(tf.concat([SS22_r_a, X2_t], axis = -1), BS ** 2)
                                         , X2 = self.tile(X11r_X1t, BS**2) 
                                         , median = self.medians_n4) # (n4)
            part2_2_2 = tf.reduce_mean(E_K_part2_2 * omega_r_t_22_2)  * gamma 
            
            part2_2 = part2_2_1 + part2_2_2
            ######
            part2_3 = self._cal_dist(X1 = self.repeat(X22r_X2t, BS ** 2), X2 = self.tile(X11r_X1t, BS ** 2), median = self.medians_n4)  # n4 # rrr_rrt - ttr_ttt
            part2_3 = tf.reduce_mean(omega_r_t_22_2 * part2_3)

            part2 = (part2_1 + part2_3 - part2_2)# / BS ** 2
            #################################### part 3 ####################################
            dxx3 = self._cal_dist(X1 = self.repeat(XX1, BS), X2 = self.tile(XX2, BS), median = self.medians_n2) # n2
            part3 = (1 - gamma) ** 2 * tf.reduce_mean(dxx3) 
            ##################################### final loss ######################################################
            loss = (part1 + part2 + part3) * 1e3
            
            return loss # tf.abs(loss)
    ########################################################################################################
    def fit(self, batch_size=32, gamma=0.99, max_iter=100, print_freq = 5, tolerance = 5, rep_loss = 5):
        self.medians_n2, self.medians_n3, self.medians_n4 = self._compute_medians()
        self.BS = batch_size
        cnt_tolerance = 0
        opt_loss = 1e10
        n = batch_size
        with tf.device('/gpu:' + str(self.gpu_number)):
            """ updated from (-1) to (-1,); [12/22]; not sure why; GPU issues"""
            self.del_dup_n3 = tf.cast(tf.repeat(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1,)), n), tf.float64)
            self.del_dup_n4 = tf.cast(tf.repeat(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1,)), n ** 2), tf.float64) * tf.cast(tf.tile(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1,)), [n ** 2]), tf.float64)
        for i in range(max_iter):
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                loss = 0
                ### mean  
                for j in range(rep_loss):
                    with tf.device('/gpu:' + str(self.gpu_number)):
                        transitions = self.replay_buffer.sample(batch_size)
                        S1, A1, _ = transitions[0], transitions[1], transitions[3]
                        
                        transitions = self.replay_buffer.sample(batch_size)
                        S2, A2, _ = transitions[0], transitions[1], transitions[3]

                        transitions = self.replay_buffer.sample(batch_size)
                        S11, A11, SS11 = transitions[0], transitions[1], transitions[3]

                        transitions = self.replay_buffer.sample(batch_size)
                        S22, A22, SS22 = transitions[0], transitions[1], transitions[3]

                        ### mean
                        transitions = self.replay_buffer.sample(min(batch_size * 10, self.replay_buffer.N))
                        _S, _A = transitions[0], transitions[1]                    
                        X_t = self.tile(tf.concat([S1, A1[:,np.newaxis]], axis=-1), len(_S))
                        X_t2 = self.tile(tf.concat([S2, A2[:,np.newaxis]], axis=-1), len(_S))
                        X_r = self.repeat(tf.concat([_S, _A[:,np.newaxis]], axis=-1), batch_size)
                        Xr_Xt = tf.concat([X_r, X_t], axis=-1) # n2
                        Xr_Xt2 = tf.concat([X_r, X_t2], axis=-1) # n2
                        omega_r_t = self.model.call(Xr_Xt) # n2
                        omega_r_t2 = self.model.call(Xr_Xt2) # n2
                        mean_omega = tf.reduce_mean(tf.reshape(omega_r_t, [batch_size * 10, batch_size]), 0)
                        mean_omega2 = tf.reduce_mean(tf.reshape(omega_r_t2, [batch_size * 10, batch_size]), 0)

                        ### mean
                    loss += self._compute_loss(S1, A1, S2, A2
                                               , S11, A11, SS11, S22, A22, SS22
                                               , gamma, mean_omega, mean_omega2)
                loss /= rep_loss
            with tf.device('/gpu:' + str(self.gpu_number)):
                dw = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
                self.losses.append(loss.numpy())
            if i % 5 == 0 and i >= 10:
                if i >= 50:
                    mean_loss = np.mean(self.losses[(i - 50):i])
                else:
                    mean_loss = np.mean(self.losses[(i - 10):i])
                if i % print_freq == 0:
                    print("omega_SASA training {}/{} DONE! loss = {:.5f}".format(i, max_iter, mean_loss))
                if mean_loss / opt_loss - 1 > -0.01:
                    cnt_tolerance += 1
                if mean_loss < opt_loss:
                    opt_loss = mean_loss
                    cnt_tolerance = 0
                if mean_loss < 0 or mean_loss < self.losses[0] / 10:
                    break
            elif i < 10 and i % 5 == 0 and print_freq < 200:
                print("omega_SASA training {}/{} DONE! loss = {:.5f}".format(i, max_iter, loss.numpy()))
                # print("loss for iter {} = {:.2f}".format(i, loss.numpy()))
            if cnt_tolerance >= tolerance:
                break 

        self.model(np.random.randn(2, self.model.input_dim))
    ########################################################################################################
    def predict(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                return np.vstack([self.model.call(inputs).cpu().numpy() for inputs in input_batches])
            else:
                return self.model.call(inputs).cpu().numpy() 
##############################################################################################################################

class Omega_SASA_Model(tf.keras.Model):
    ''' weights = self.model(S_r, A_r, S_t)
    initial? a random different?
    
    Input for `VisitationRatioModel()`; NN parameterization
    '''
    def __init__(self, S_dims, h_dims, A_dims = 1, gpu_number = 0):
        super(Omega_SASA_Model, self).__init__()
        self.hidden_dims = h_dims
        self.gpu_number = gpu_number

        self.input_dim = S_dims + A_dims + S_dims + A_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")

            self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")

            self.w21 = self.xavier_var_creator(self.input_shape2, name = "w21")
            self.b21 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b21")

            self.w22 = self.xavier_var_creator(self.input_shape2, name = "w22")
            self.b22 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b22")

            self.w3 = self.xavier_var_creator(self.input_shape3, name = "w3")
            self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64), name = "b3")


    def xavier_var_creator(self, input_shape, name = "w3"):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0) / 3
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var

    def call(self, inputs):
        """
        inputs: concatenations of S, A, S_t, A_t = [r,r,t]
        outputs: weights?
        where do we build the model?
        parameter for the depth!!!
        """
        z = tf.cast(inputs, tf.float64)
        h1 = tf.nn.leaky_relu(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        h2 = tf.nn.relu(tf.matmul(h2, self.w21) + self.b21)
        h2 = tf.nn.relu(tf.matmul(h2, self.w22) + self.b22)
        out = (tf.matmul(h2, self.w3) + self.b3) 
        out = tf.math.log(1.001 + tf.exp(out))
#         out = tf.clip_by_value(out, 0.5, 2)
#         out = tf.clip_by_value(out, 0.1, 10)
#         out = tf.clip_by_value(out, 0.01, 100)
        return out


    def predict_4_VE(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                return np.vstack([tf.identity(self.call(dat)).numpy() for dat in input_batches])
            
            else:
                return tf.identity(self.call(inputs)).numpy()

#                 return np.vstack([self.call(inputs).cpu().numpy() for inputs in input_batches])
#             else:
#                 return self.call(inputs).cpu().numpy() 

    
    #     def _normalize(self, weights, BS):
#         """
#         check !!!
#         """
#         weights = tf.reshape(weights, [BS, BS])
#         weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
#         weights = weights / weights_sum
#         return tf.reshape(weights, [BS**2])
        


# omega_r_t_true = tf.cast(tf.cast((Xr_Xt[:, 0] > 0), np.float64) * self.A_factor_over_S == Xr_Xt[:, 2], np.float64) * 2

# #################################### part 1, n3 #################################### 
# dxx_ra_t_2_2 = self.repeat(tf.concat([S_r_a, X_t], axis = -1), BS) - self.tile(XX, BS ** 2) # (n3), rr_rt - tt_tt
# E_K_ra_t_2_2 = self._cal_dist(X1_X2 = dxx_ra_t_2_2, median = self.medians_n3)

# K_r_t_2_2 = self._cal_dist(X1 = self.repeat(Xr_Xt, BS), X2 = self.tile(XX, BS ** 2), median = self.medians_n3)

# part1_bef_sum = tf.squeeze(self.repeat(omega_r_t_true, BS)) * (gamma * E_K_ra_t_2_2 - K_r_t_2_2) * self.del_dup_n3
# part1 = 2 * (1 - gamma) * tf.reduce_mean(part1_bef_sum) 
# #################################### part 2, n4 #################################### 
# omega_r_t_22_2 = tf.squeeze(self.repeat(omega_r_t_true, BS ** 2)) * tf.squeeze(self.tile(tf.expand_dims(omega_r_t_true, 1), BS ** 2)) * self.del_dup_n4 # n4
# ######
# K_term = self._cal_dist(X1 = self.tile(tf.concat([S_r_a, X_t], axis = -1), BS ** 2)
#                       , X2 = tf.repeat(tf.concat([S_r_a, X_t], axis = -1), BS ** 2, axis=0)
#                       , median = self.medians_n4)
# part2_1 = gamma ** 2 * tf.reduce_mean(K_term * omega_r_t_22_2)
# ######
# diff_part2_2 = self.repeat(tf.concat([S_r_a, X_t], axis = -1), BS ** 2) - self.tile(Xr_Xt, BS**2) # (n4) # rrr_rrt - ttr_ttt
# E_K_part2_2 = self._cal_dist(X1_X2 = diff_part2_2, median = self.medians_n4) # (n4)

# part2_2 = tf.reduce_mean(E_K_part2_2 * omega_r_t_22_2) * 2 * gamma 
# ######
# part2_3 = self._cal_dist(X1 = self.repeat(Xr_Xt, BS ** 2), X2 = self.tile(Xr_Xt, BS ** 2), median = self.medians_n4)  # n4 # rrr_rrt - ttr_ttt
# part2_3 = tf.reduce_mean(omega_r_t_22_2 * part2_3)

# part2 = (part2_1 + part2_3 - part2_2)# / BS ** 2
# #################################### part 3 ####################################
# dxx3 = self._cal_dist(X1 = self.repeat(XX, BS), X2 = self.tile(XX, BS), median = self.medians_n2) # n2
# part3 = (1 - gamma) ** 2 * tf.reduce_mean(dxx3) 
# ##################################### final loss ######################################################
# loss_true = (part1 + part2 + part3) * 1e3
# print(loss - loss_true)

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.backend.set_floatx('float64')
# state_ratio_model = VisitationRatioModel(model, optimizer, replay_buffer,
#                  target_policy, behavior_policy, medians=None)
# state_ratio_model.fit(BS=32, gamma=0.99, max_iter=100)
# state_ratio = state_ratio_model.predict(state; S, A)


class VisitationRatioModel_init_SA():
    """ w(S, A) <- model
    """
    def __init__(self, replay_buffer
                 , target_policy, A_range
                 , h_dims, sepe_A = 0
                 , lr = 0.0005, behav = None
                 , A_dims = 1, gpu_number = 0
                 , w_clipping_val = 0.5, w_clipping_norm = 1.0, beta_1 = 0.5
                 , optimizer = None, medians=None):
        
        self.model = Omega_SA_Model(replay_buffer.S_dims, h_dims, A_dims, gpu_number)
        self.S_dims = replay_buffer.S_dims
        self.A_dims = A_dims
        self.gpu_number = gpu_number
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.A_range = A_range
        self.losses = []
        self.sepe_A = sepe_A
        
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                lr, decay_steps = 100, decay_rate = 0.99
                                                )
                                                  , clipnorm = w_clipping_norm, clipvalue = w_clipping_val)
        # self.optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta_1, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    def _compute_medians(self, n= 32, rep = 20):
        # to save memory, we do it iteratively
        # medians_11_22, medians_12_33, medians_12_34 = [tf.zeros([(self.S_dims + self.A_dims)], tf.float64) for _ in range(3)]
        with tf.device('/gpu:' + str(self.gpu_number)):
            
            if self.sepe_A:
                median = tf.zeros([(self.S_dims)], tf.float64)
            else:
                median = tf.zeros([(self.S_dims + self.A_dims)], tf.float64)
            # median = 0

            for i in range(rep):
                transitions = self.replay_buffer.sample(n)
                S, A = transitions[0], transitions[1] 
                dSS = tf.repeat(S, n, axis=0) - tf.tile(S, [n, 1])
                median_SS = tf.reduce_mean(tf.math.abs(dSS), axis=0)
                if self.sepe_A:
                    median += median_SS + 1e-6
                else:
                    dAA = tf.repeat(A[:,np.newaxis], n, axis=0) - tf.tile(A[:,np.newaxis], [n, 1])
                    median_AA = tf.reduce_mean(tf.math.abs(dAA), axis=0) #/ 10
                    median += tf.concat([median_SS, median_AA], axis = 0) + 1e-6
#                     X = tf.concat([S, A[:,np.newaxis]], axis=-1)
#                     dxx = tf.repeat(X, n, axis=0) - tf.tile(X, [n, 1])
#                 median += np.mean(tf.math.abs(dxx), axis=0) + 1e-2
#                 median += np.median(tf.reduce_sum(tf.math.abs(dxx), axis = -1)) + 1e-2

            median = median / rep #medians_11_22 / rep, medians_12_33 / rep, medians_12_34 / rep
            return median * (self.S_dims + self.A_dims)
    

    def _cal_dist(self, X1 = None, X2 = None, X1_X2 = None, median = None):
        """
        Laplacian Kernel
        pairwise difference: Delta. exp(-diff)
        """
        # shared median
#         if X1_X2 is not None:
#             return tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2), axis=-1) / median) 
#         else:
#             return tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=-1) / median)
        if not self.sepe_A:
            if X1_X2 is not None:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2) / median, axis=-1)) 
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2) / median, axis=-1))
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
        else:
            if X1_X2 is not None:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]) / median, axis=-1))  * tf.cast(X1_X2[:, self.S_dims] == 0, tf.int32)
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]) / median, axis=-1)) * tf.cast(X1[:, self.S_dims] == X2[:, self.S_dims], tf.float64)
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
  
    def repeat(self, X, rep):
        return tf.repeat(X, rep, axis=0)
    def tile(self, X, rep):
        return tf.tile(X, [rep, 1])
    
    def _compute_loss(self, S, A, SS
                     , S2, A2, SS2):
        """n2 = n^2
        
        r left, t right...
        
        _r = repeat = 11
        _t = tile = 1
        order? 
        repeat = 11; tile = 22
        X = [S, A]
        XX = [X, X]
        BS = batch_size
        """
        
        A_factor_over_S = 10
        A *= A_factor_over_S
        A2 *= A_factor_over_S
        
        with tf.device('/gpu:' + str(self.gpu_number)):
            BS = self.BS
            ###########################################################################
            X = tf.concat([S, A[:,np.newaxis]], axis=-1)
            X2 = tf.concat([S2, A2[:,np.newaxis]], axis=-1)
            X_r = self.repeat(X, BS)  # n2 x ...
            X_t = self.tile(X, BS) # n2 x ...
            X2_t = self.tile(X2, BS) # n2 x ...
            S_r = self.repeat(S, BS) # n2 x ...
            S_t = self.tile(S, BS) # n2 x ...
            SS_r = self.repeat(SS, BS) # n2 x ...
            SS_t = self.tile(SS, BS) # n2 x ...        
            
            ####################################
            A_of_S = tf.convert_to_tensor(self.target_policy.get_A(S), dtype=tf.float64)
            A_of_S *= A_factor_over_S
            A_of_S2 = tf.convert_to_tensor(self.target_policy.get_A(S2), dtype=tf.float64)
            A_of_S2 *= A_factor_over_S
            S_a = tf.concat([S, A_of_S[:, np.newaxis]], axis = -1)
            S2_a = tf.concat([S2, A_of_S2[:, np.newaxis]], axis = -1)
            S_r_a = self.repeat(S_a, BS)
            S_t_a = self.tile(S_a, BS)
            S2_t_a = self.tile(S2_a, BS)
            
            A_of_SS = tf.convert_to_tensor(self.target_policy.get_A(SS), dtype=tf.float64)
            A_of_SS *= A_factor_over_S
            A_of_SS2 = tf.convert_to_tensor(self.target_policy.get_A(SS2), dtype=tf.float64)
            A_of_SS2 *= A_factor_over_S

            SS_a = tf.concat([SS, A_of_SS[:, np.newaxis]], axis = -1)
            SS2_a = tf.concat([SS2, A_of_SS2[:, np.newaxis]], axis = -1)
            SS_r_a = self.repeat(SS_a, BS)
            SS2_t_a = self.repeat(SS2_a, BS)
            SS_t_a = self.tile(SS_a, BS)
    
            ####################################
            ############ omega ############
            omega = self.model.call(X) # n
            omega = omega / self.mean_omega #tf.reduce_mean(omega)
            omega = tf.squeeze(omega)
            omega_t = tf.squeeze(tf.tile(omega[:, np.newaxis], [BS, 1]))
            
            omega2 = self.model.call(X2) # n
            omega2 = omega2 / self.mean_omega #tf.reduce_mean(omega)
            omega2 = tf.squeeze(omega2)
            omega2_t = tf.squeeze(tf.tile(omega2[:, np.newaxis], [BS, 1]))

            #################################### part 1 #################################### 
            K_1 = tf.reduce_mean(self._cal_dist(X1 = SS_r_a, X2 = S2_t_a, median = self.median) * self.repeat(omega, BS))
            K_1 *= (2 * self.gamma * (1 - self.gamma)) #* self.gamma

            K_2 = self._cal_dist(X1 = X_r, X2 = S2_t_a, median = self.median) * self.repeat(omega, BS) #* omega_t
            K_2 = tf.reduce_mean(K_2) 
            K_2 *= (2 * (1 - self.gamma))

            part1 = K_1 - K_2
            #################################### part 2 #################################### 
            omega_r_t = self.repeat(omega, BS) * tf.squeeze(self.tile(omega2[:, tf.newaxis], BS)) # n2
            ########
            part2_1 = tf.reduce_mean(self._cal_dist(X1 = SS_r_a, X2 = SS2_t_a, median = self.median) * omega_r_t)
            part2_1 *= self.gamma ** 2
            ########
            part2_3 = self._cal_dist(X1 = X_r, X2 = X2_t, median = self.median)  # n4 # rrr_rrt - ttr_ttt
            part2_3 = tf.reduce_mean(omega_r_t * part2_3)
            ########
            part2_2 = tf.reduce_mean(self._cal_dist(X1 = SS_r_a, X2 = X2_t, median = self.median) * omega_r_t)
            part2_2 *= (2 * self.gamma)

            part2 = part2_1 - part2_2 + part2_3
            #################################### part 3 ####################################
            part3 = tf.reduce_mean(self._cal_dist(X1 = S_r_a, X2 = S2_t_a, median = self.median))
            part3 *= (1 - self.gamma) ** 2
            ##################################### final loss ######################################################
            loss = part1 + part2 + part3
            return loss * 10000
    ############################################################################################################################################################################################################################################################################################################################################################################################################
    def fit(self, batch_size=32, gamma=0.99, max_iter=100, print_freq = 20, tolerance = 5, rep_loss = 3):
        self.gamma = gamma
        self.median = self._compute_medians()
        # , self.medians_n3, self.medians_n4
        self.BS = batch_size
        cnt_tolerance = 0
        opt_loss = 1e10
        for i in range(max_iter):
            ##### compute loss function #####
            with tf.device('/gpu:' + str(self.gpu_number)):
                with tf.GradientTape() as tape:
                    loss = 0
                    ###
                    transitions = self.replay_buffer.sample(min(batch_size * 10, self.replay_buffer.N))
                    S, A, SS = transitions[0], transitions[1], transitions[3]
                    X = tf.concat([S, A[:,np.newaxis]], axis=-1)
                    omega = self.model.call(X) # n
                    self.mean_omega = tf.reduce_mean(omega)
                    ###
                    for j in range(rep_loss):
                        transitions = self.replay_buffer.sample(batch_size)
                        S, A, SS = transitions[0], transitions[1], transitions[3]
                        
                        # tilde
                        transitions = self.replay_buffer.sample(batch_size)
                        S2, A2, SS2 = transitions[0], transitions[1], transitions[3]

                        loss += self._compute_loss(S, A, SS, S2, A2, SS2)        
                    loss /= rep_loss

                dw = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
                self.losses.append(loss.numpy())
                
            if i % 5 == 0 and i >= 10:
                if i >= 50:
                    mean_loss = np.mean(self.losses[(i - 50):i])
                else:
                    mean_loss = np.mean(self.losses[(i - 10):i])
                if mean_loss / opt_loss - 1 > -0.01:
                    cnt_tolerance += 1
                if mean_loss < opt_loss:
                    opt_loss = mean_loss
                    cnt_tolerance = 0
                if mean_loss < 0 or mean_loss < self.losses[0] / 10:
                    break
            if i % print_freq == 0 and i >= 10:
                print("omega_SA training {}/{} DONE! loss = {:.5f}".format(i, max_iter, mean_loss))
            if cnt_tolerance >= tolerance:
                break 

        self.model(np.random.randn(2, self.model.input_dim))
        
    def predict(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        if inputs.shape[0] > batch_size:
            n_batch = inputs.shape[0] // batch_size + 1
            input_batches = np.array_split(inputs, n_batch)
            return np.vstack([self.model.call(inputs).cpu().numpy() for inputs in input_batches])
        else:
            return self.model.call(inputs).cpu().numpy() 
######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class Omega_SA_Model(tf.keras.Model):
    def __init__(self, S_dims, h_dims, A_dims = 1, gpu_number = 0):
        super(Omega_SA_Model, self).__init__()
        self.gpu_number = gpu_number
        self.hidden_dims = h_dims
        self.seed = 42
        self.input_dim = S_dims + A_dims
        self.reset()
        
    def reset(self):
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.input_shape2 = [self.hidden_dims, self.hidden_dims]
            self.input_shape3 = [self.hidden_dims, 1]

            self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")

            self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")

            self.w21 = self.xavier_var_creator(self.input_shape2, name = "w21")
            self.b21 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b21")

            self.w22 = self.xavier_var_creator(self.input_shape2, name = "w22")
            self.b22 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b22")
            
            self.w23 = self.xavier_var_creator(self.input_shape2, name = "w23")
            self.b23 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b23")
            
            self.w24 = self.xavier_var_creator(self.input_shape2, name = "w24")
            self.b24 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b24")

            self.w3 = self.xavier_var_creator(self.input_shape3, name = "w3")
            self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64), name = "b3")

    def xavier_var_creator(self, input_shape, name = "w3"):
        tf.random.set_seed(self.seed)
        
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0) / 5
        
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var

    def call(self, inputs):
        """
        inputs are concatenations of S, A, S_t, A_t = [r,r,t]
        """
        z = tf.cast(inputs, tf.float64)
        h1 = tf.nn.leaky_relu(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        h2 = tf.nn.relu(tf.matmul(h2, self.w21) + self.b21)
        h2 = tf.nn.relu(tf.matmul(h2, self.w22) + self.b22)
        
        h2 = tf.nn.relu(tf.matmul(h2, self.w23) + self.b23)
        h2 = tf.nn.relu(tf.matmul(h2, self.w24) + self.b24)


        out = (tf.matmul(h2, self.w3) + self.b3) #/ 100 # o.w., the initialization will die
        out = tf.math.log(1.0001 + tf.exp(out))
#         out = tf.clip_by_value(out, 0.1, 10)
        return out

    def predict_4_VE(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                #return np.vstack([self.call(dat).cpu().numpy() for dat in input_batches])
                return np.vstack([tf.identity(self.call(dat)).numpy() for dat in input_batches])
            
            else:
                return tf.identity(self.call(inputs)).numpy()
                #return self.call(inputs).cpu().numpy() 

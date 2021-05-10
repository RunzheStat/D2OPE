from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from _util import *

import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_probability as tfp
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from coinDice import common as common_utils
from coinDice import estimator as estimator_lib
from coinDice.dataset import Dataset, EnvStep, StepType
from coinDice.env_policies import get_target_policy
from coinDice.value_network import ValueNetwork
from coinDice import  neural_coin_dice 
from coinDice import converter

reload(neural_coin_dice)

######################################################################################################################################################
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

######################################################################################################################################################

def coindice(dataset, pi1, N, T, gamma = .95, hidden_dims = (64,), gpu_number = 0
             , nu_learning_rate = 0.01, zeta_learning_rate = 0.01, weight_learning_rate = 0.01
             , nu_regularizer = 0, zeta_regularizer = 0
             , num_steps = 20000, batch_size = 512, alpha_CI = 0.9
             , print_loss = False, print_freq = 500, print_prog = True, patience = 5
            , primal_form = True, f_exponent = 1.5, algae_alpha = 0.01):
    ################################################################################################################################################
    from scipy.stats import chi2
    divergence_limit = chi2.ppf(alpha_CI, 1) / N / T
    n_intervals = 1

    ################################################################################################################################################
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        stddev=0.5, seed=1)
    with tf.device('/gpu:' + str(gpu_number)):
        ##############################################################################
        nu_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                                  fc_layer_params=hidden_dims,
                                  activation_fn=activation_fn,
                                  kernel_initializer=kernel_initializer,
                                  last_kernel_initializer=None,
                                  output_dim = 2 * 2 * n_intervals) 
        zeta_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                                    fc_layer_params=hidden_dims,
                                    activation_fn=activation_fn,
                                    kernel_initializer=kernel_initializer,
                                    last_kernel_initializer=None,
                                    output_dim=2 * 2 * n_intervals)
        weight_network = ValueNetwork((dataset.spec.observation,  # initial state
                                       dataset.spec.observation,  # cur state
                                       dataset.spec.action,       # cur action
                                       dataset.spec.observation), # next state
                                      fc_layer_params=hidden_dims,
                                      activation_fn=activation_fn,
                                      kernel_initializer=kernel_initializer,
                                      last_kernel_initializer=None,
                                      output_dim=2 * n_intervals)

        nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, beta_2=0.99)
        zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, beta_2=0.99)
        weight_optimizer = tf.keras.optimizers.Adam(weight_learning_rate, beta_2=0.99)
        ################################################################################################################################################
        estimator = neural_coin_dice.NeuralCoinDice(dataset.spec,
                                   nu_network, zeta_network, weight_network,
                                   nu_optimizer, zeta_optimizer, weight_optimizer,
                                   gamma=gamma, gpu_number = gpu_number
                                   , divergence_limit=divergence_limit,
                                   f_exponent=f_exponent,
                                   primal_form=primal_form, divergence_type = 'kl',
                                   nu_regularizer=nu_regularizer,
                                   zeta_regularizer=zeta_regularizer,
                                   algae_alpha=algae_alpha * np.array([1, 1]),
                                   unbias_algae_alpha=False,
                                   closed_form_weights=True,
                                   num_samples=None)

        global_step = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(global_step)
        ################################################################################################################################################
        def one_step(env_steps, next_steps, initial_steps_batch):
            global_step.assign_add(1)
            with tf.summary.record_if(tf.math.mod(global_step, 25) == 0):
                losses, _ = estimator.train_step(initial_steps_batch, env_steps, next_steps,
                                                 pi1)
            return losses
        ################################################################################################
        summary_writer = tf.summary.create_noop_writer()
        stop_cnt = 0
        cret_idx = 1
        with summary_writer.as_default():
            running_losses = []
            running_estimates = []
            optimal_loss = np.repeat(1e10, 4)
            for step in range(num_steps):
                env_steps, next_steps = dataset.sample_step_and_next(batch_size)
                initial_steps_batch = dataset.sample_init_steps(batch_size)
                losses = one_step(env_steps, next_steps, initial_steps_batch)

                running_losses.append([t.numpy() for t in losses])
                estimate = np.mean(running_losses, 0)[0] # the CI
                running_estimates.append(estimate)
                
                
                if step % print_freq == 0 or step == num_steps - 1:
                    if print_prog:
                        print('step', step)
                    mean_loss = np.mean(running_losses, 0)[1:] # do not include the estimate
                    mean_loss = np.array([np.mean(a ** 2) for a in mean_loss])
                    if mean_loss[cret_idx] > optimal_loss[cret_idx] and step > 500:
                        stop_cnt += 1
#                         print(stop_cnt, mean_loss[cret_idx], optimal_loss[cret_idx])
                    if stop_cnt == patience:
                        break
                    if mean_loss[cret_idx] < optimal_loss[cret_idx] and step > 500:
                        optimal_loss[cret_idx] = mean_loss[cret_idx]
                        stop_cnt = 0
                    # print(optimal_loss[cret_idx], stop_cnt)

                    if print_loss:
                        print(mean_loss) # weighted_nu_loss, weighted_zeta_loss, weight_loss, divergence
                        # print('losses', np.mean(running_losses, 0)[1:])
                    
                    for idx, est in enumerate(estimate):
                        tf.summary.scalar('estimate%d' % idx, est)
                    
                    # print('estimated confidence interval %s' % (np.array(estimate) / (1 - gamma) ))
                    if print_prog:
                        print('avg last 50 estimated confidence interval %s' %
                              (np.mean(running_estimates[-50:], axis=0) / (1 - gamma)))
                    running_losses = []
    return estimate / (1 - gamma)
    ################################################################################################
    # if save_dir is not None:
    #     results_filename = os.path.join(save_dir, 'results.npy')
    #     with tf.io.gfile.GFile(results_filename, 'w') as f:
    #         np.save(f, running_estimates)

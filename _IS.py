from _util import *

def cal_Bernstein_CI(Ys, V_min = -100, V_max = 0):
    # upper?
    b = 1
    CIs = []
    n = len(Ys)
    Ys = (Ys - V_min) / (V_max - V_min)
    u = np.mean(Ys)
    pair_variance = 0
    for i in range(n):
        for j in range(n):
            pair_variance += (Ys[i] - Ys[j]) ** 2
    for delta in [0.5, 0.025]:
        log_term = log(2 / delta)
        pair_variance_term = np.sqrt(pair_variance * log_term / (n - 1))
        lower = u - (7 * b / 3) * log_term / (n - 1) - 1 / n * pair_variance_term
        upper = u + (7 * b / 3) * log_term / (n - 1) + 1 / n * pair_variance_term
        lower = lower * (V_max - V_min) + V_min
        upper = upper * (V_max - V_min) + V_min
        CIs.append([lower, upper])
    return CIs # [90%, 95%]

def cal_step_IS(trajs_train_resp, gamma, pi_behva, pi1):
    V_seeds = []
    seed = 0
    for trajs in trajs_train_resp:
        V_trajs = []
        V_max = np.max(([[step[2] for step in traj] for traj in trajs])) / (1 - gamma)
        V_min = np.min(([[step[2] for step in traj] for traj in trajs])) / (1 - gamma)
        for traj in trajs:
            w_this_traj = []
            As = [step[1] for step in traj]
            As = arr(As).astype(np.int)
            Rs = arr([step[2] for step in traj])
            ########################
            Ss = arr([step[0] for step in traj])
            behav_probs = pi_behva.get_A_prob(Ss)
            behav_A_probs = behav_probs[range(len(behav_probs)), As]
            tp_probs = pi1.get_A_prob(Ss)
            tp_A_probs = tp_probs[range(len(tp_probs)), As]
            r_this = tp_A_probs / behav_A_probs 
            ########################
            for t in range(1, len(traj) + 1):
                w_this_traj.append(multiplyList((tp_A_probs / behav_A_probs)[:t]))
            w_Rs = arr(w_this_traj) * Rs
            V = sum(w_Rs[t] * gamma ** t for t in range(len(w_Rs)))
            V_trajs.append(V)
        V_seeds.append([np.array(V_trajs), V_min, V_max])
        seed += 1
    return V_seeds
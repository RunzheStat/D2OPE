from _util import *
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import BlendedGenericTransform
########################################################################################################################################################################
def get_tableau20():
    # These are the "Tableau 20" colors as RGB.   
    # , (174, 199, 232)
    tableau20 = [(31, 119, 180), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
    return tableau20

def plot_curves(values, horizontal, x_axis, title = None, y_lab = "coverage frequency"
              , labels = ["DR", "TR", "QR"], legend_title = "method"
              , y_low = None, y_high = None, marker = None
              , xlabel = "tau", path = None, adjust_color = None
                , fake_line = None
                , title_size = 18, x_label_size = 14, y_label_size = 14
              , ax = None, is_sub_plot = False, print_legend = True
               ):
    #####################
    rc('mathtext', default='regular')
    tableau20 = get_tableau20()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    #####################    
    color = tableau20[::2]
    i = 0
    ############################# ?? ##############################
    legends = []
    for i in range(len(labels)):
        if adjust_color is not None and i in adjust_color:
            legends += ax.plot(x_axis, values[i], '-', label = labels[i], color = adjust_color[i], marker = marker)
        else:
            legends += ax.plot(x_axis, values[i], '-', label = labels[i], color = color[i], marker = marker)
    ###############################################################
    if horizontal is not None:
        ax.hlines(y = horizontal, xmin = x_axis[0]
              , xmax=x_axis[-1] #, label = "nominal"
              , color= "slategrey" #tableau20[11]
             , linestyles = "dashed", linewidth=3)
#     if fake_line is not None:
#         legends += ax.plot(x_axis, fake_line["value"], '-', label = fake_line["label"], color = fake_line["color"], marker = marker)
    ax.grid()
    ###############################################################
    if title is not None:
        ax.set_title(title, fontsize = title_size)
    ax.set_xlabel(xlabel, fontsize = x_label_size)
    ax.set_ylabel(y_lab, fontsize = y_label_size) # , color= "red"
#     ax.yaxis.set_label_coords(-0.1,0.5)
    ###############################################################
    if print_legend:
        ax.legend(legends, labels, title = legend_title, loc=0)
    #####################
    ax.set_xticks(ticks = x_axis)
    if y_low is None:
        y_low = np.min(values)
    if y_high is None:
        y_high = np.min(y_high)
    ax.set_ylim(bottom = y_low, top = y_high)     
    #####################
    if is_sub_plot:
        return ax, legends
    else:
        if path is not None:
            fig.savefig(path)
        return fig

def combine_two_part(r1, r2, setting):
    r = {}
    r["freq"] = (r2["freq"] + r1["freq"]) / 2
    r['RMSE'] = np.sqrt((r2["RMSE"] ** 2 + r1["RMSE"] ** 2) / 2)
    r["bias"] = (arr([mean(r1[est]["error"]) for est in ["DR", "TR", "QR"]]) + arr([mean(r2[est]["error"]) for est in ["DR", "TR", "QR"]])) / 2
    r["V_true"] = r2["V_true"]
    r["raw_Q"] = np.append(r1["raw_Q"], r2["raw_Q"])
    dump(r, "res/" + setting)
    
def get_std_of_RMSE(errors):
    sampled_RMSEs = zeros(1000)
    for i in range(1000):
        sampled_errors = np.random.choice(errors, len(errors))
        sampled_RMSE = np.sqrt(np.mean(sampled_errors ** 2))
        sampled_RMSEs[i] = sampled_RMSE
    return std(sampled_RMSEs)
    
def get_width(r, log_width = True, mean_width = True, CI_idx = 1, coindice = False):
    if coindice:
        widths = [a[1] - a[0] for a in r['CIs']]
        # can be even negative
        widths = np.clip(widths, 0.01, None)
    else:
        widths = [(a[CI_idx][1] - a[CI_idx][0]) for a in r['all_CI']]
        widths = arr(widths)
    if log_width:
        widths = np.log(widths)
    if mean_width:
        return np.mean(widths)
    
def extract_our_res(r, delete_IS = False, log_width = True):
    if log_width:
        """ mistake here 
        width[1:],
        """
        width = arr([np.mean(np.log(arr(r[est]["stds"]) * 1.96 * 2)) for est in ["DR", "TR", "QR"]])
    Q_errors = arr(r['raw_Q']) - r['V_true'][0]
    RMSE_Q = np.sqrt(np.mean(Q_errors ** 2))
    bias_Q = mean(Q_errors)
    RMSE_this = np.append(r['RMSE'], [RMSE_Q])
    """ no IS here """
    bias_this = [mean(r[est]["error"]) for est in ["DR", "TR", "QR"]]
    bias_this.append(bias_Q)
    f095_this = list(r['freq'][0])
    f09_this = list(r['freq'][1])

    if delete_IS:
        return f09_this[1:], f095_this[1:], width, bias_this, RMSE_this[1:]
    else:
        return f09_this, f095_this, width, bias_this, RMSE_this

def extract_std(r, delete_IS = False, log_width = True):
    N = len(r['DR']["stds"])
    sqrtN = np.sqrt(N)
    if log_width:
        width = arr([np.std(np.log(arr(r[est]["stds"]) * 1.96 * 2)) for est in ["DR", "TR", "QR"]])
        width /= sqrtN
    ###
    Q_errors = arr(r['raw_Q']) - r['V_true'][0]
    bias_Q = std(Q_errors) / sqrtN    
    bias_this = [np.std(r[est]["error"]) / sqrtN for est in ["DR", "TR", "QR"]]
    bias_this.append(bias_Q)

    """ NOT YET """
    RMSE_Q = get_std_of_RMSE(Q_errors)
    RMSE_this = np.append([get_std_of_RMSE(r[est]['error']) for est in ["DR", "TR", "QR"]], [RMSE_Q])

    if delete_IS:
        return width, bias_this, RMSE_this[1:]
    else:
        return width, bias_this, RMSE_this

    
def multi_plots(values, names, labels = None, skylines = None):
    n_plots = len(values)
    if labels is None:
        labels = names
    
    rc('mathtext', default='regular')
    tableau20 = get_tableau20()
    
    N_iter = len(values[0])
    iters = np.arange(N_iter)
    fig = plt.figure()
    plt.figure(figsize=(12, 2))    
    plt.subplots_adjust(right = 1.4)
    for i in range(n_plots):
        ax = plt.subplot(1, n_plots, i + 1)
        plt.title(names[i], fontdict = {"fontsize" : 10})
        plt.xticks(ticks = np.arange(1, N_iter + 2, 1))
        lns1 = ax.plot(iters, values[i], '-', label = names[i], color = tableau20[i])
        
        if skylines and skylines[i]:
            ax.hlines(y=skylines[i], xmin=iters[0], xmax=iters[-1], color=tableau20[15])


        ax.grid()
        ax.set_xlabel("Iteration")
        ax.set_ylabel(labels[i]) # , color= "red"

    plt.show()


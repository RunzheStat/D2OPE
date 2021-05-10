from _util import *
###########################################################################################################################################################################################################################################################################################################################

class recorder():
    def __init__(self):
        # length + coverage frequency
        self.IS = { "error" : []
              , "stds" : []
              , "freq" : {"5" : [], "10" : []}
             }
        self.DR = { "error" : []
              , "stds" : []
              , "freq" : {"5" : [], "10" : []}
             }
        self.TR = {"error" : []
              , "stds" : []
              , "freq" : {"5" : [], "10" : []}}
        self.QR = {"error" : []
              , "stds" : []
              , "freq" : {"5" : [], "10" : []}}
        self.raw_Q = []
        self.V_true = []
        self.seed = 0
        self.instances = []
        self.names = ["IS", "DR", "TR", "QR"]
    
    def add_env(self, fqi, fqe):
        self.fqi_para = fqi
        self.fqe_para = fqe


    def update(self, V_true, are = None, are_details = None, dis = False, prec = 2):
        if are_details is not None:
            raw_Qs, IS_V, DR_V, TR_V, QR_V = are_details
        else:
            raw_Qs, IS_V, DR_V, TR_V, QR_V = are.raw_Qs, are.IS_V, are.DR_V, are.TR_V, are.QR_V
            are.large = []
        self.seed += 1
        ############################################################################################################################################
        if dis:
            printR("true value: {:.2f} ".format(V_true))
            printR("raw Q-value: {:.2f}".format(np.mean(raw_Qs)))

            pd.set_option('precision', prec)
            printR("IS: est = {:.2f}, sigma = {:.2f}".format(IS_V["V"], IS_V["sigma"]))
            display(DF(IS_V["CIs"], index = ["0.05", "0.1"]))
            printR("DR: est = {:.2f}, sigma = {:.2f}".format(DR_V["V"], DR_V["sigma"]))
            display(DF(DR_V["CIs"], index = ["0.05", "0.1"]))
            printR("TR: est = {:.2f}, sigma = {:.2f}".format(TR_V["V"], TR_V["sigma"]))
            display(DF(TR_V["CIs"], index = ["0.05", "0.1"]))    
            printR("QR: est = {:.2f}, sigma = {:.2f}".format(QR_V["V"], QR_V["sigma"]))
            display(DF(QR_V["CIs"], index = ["0.05", "0.1"]))    

        ############################ Record results ############################
        self.raw_Q.append(np.mean(raw_Qs))
        self.V_true.append(V_true)

        self.IS["error"].append(IS_V["V"] - V_true)
        self.IS["stds"].append(IS_V["sigma"])
        self.DR["error"].append(DR_V["V"] - V_true)
        self.DR["stds"].append(DR_V["sigma"])
        self.TR["error"].append(TR_V["V"] - V_true)
        self.TR["stds"].append(TR_V["sigma"])
        self.QR["error"].append(QR_V["V"] - V_true)
        self.QR["stds"].append(QR_V["sigma"])

        for i, alpha in enumerate(["5", "10"]):
            self.IS["freq"][alpha].append(IS_V["CIs"][i][0] <= V_true and IS_V["CIs"][i][1] >= V_true)
            self.DR["freq"][alpha].append(DR_V["CIs"][i][0] <= V_true and DR_V["CIs"][i][1] >= V_true)
            self.TR["freq"][alpha].append(TR_V["CIs"][i][0] <= V_true and TR_V["CIs"][i][1] >= V_true)
            self.QR["freq"][alpha].append(QR_V["CIs"][i][0] <= V_true and QR_V["CIs"][i][1] >= V_true)
        self.instances.append(are)
        if dis:
            printG("<<================ Iteration {} DONE ! ================>>".format(self.seed))
            self.analyze()

    def analyze(self, prec = 3, echo = True):
        pd.set_option('precision', prec)
        mat = [[ np.sqrt(np.mean(arr(estimator["error"]) ** 2))
            , np.mean(np.abs(estimator["error"]))
            , np.mean(estimator["error"])
            , np.mean(estimator["stds"])
            #, np.mean(estimator['freq']['1'])
            , np.mean(estimator['freq']['5'])
            , np.mean(estimator['freq']['10'])] for estimator in [self.IS, self.DR, self.TR, self.QR]]
        df = DF(mat
              , columns = ["RMSE", "MAE", "bias", "ave_std", "freq: 0.95", "freq: 0.9"] #  "freq: 0.99", 
              , index = self.names)
        error_Q = (arr(self.raw_Q) - arr(self.V_true))
        RMSE_Q = np.sqrt(np.mean(error_Q ** 2))
        MAE_Q = np.mean(np.abs(error_Q))
        bias_Q = np.mean(error_Q)
        if echo:
            display(df)
            print("Q: RMSE = {:.2f}, bias = {:.2f}".format(RMSE_Q, bias_Q))
            printR("rep = {}".format(self.seed))
        return mat

    def save(self, path):
        freq = arr([[np.mean(estimator['freq'][alpha])
                    for estimator in [self.IS, self.DR, self.TR, self.QR]
                    ]
             for alpha in ["5", "10"]]) # "1", 

        res = {"DR" : self.DR, "TR" : self.TR, "QR" : self.QR, "IS" : self.IS
               , "raw_Q" : self.raw_Q
               , "V_true" : self.V_true
              , "RMSE" : arr([np.sqrt(np.mean(arr(estimator["error"]) ** 2))
                              for estimator in [self.IS, self.DR, self.TR, self.QR]])
               , "MAE" : arr([
                   np.mean(np.abs(estimator["error"]))
                   for estimator in [self.IS, self.DR, self.TR, self.QR]
                             ])
              , "std" : arr([np.mean(estimator["stds"])
                   for estimator in [self.IS, self.DR, self.TR, self.QR]])
              , "freq" : freq
              , "hyper": self.hyper}

        dump(res, path)
        
    def aggregate(self, results, prec = 3):
        n_reps = [len(res["DR"]["error"]) for res in results]
        total_rep = sum(n_reps)
        # n_reps = arr(n_reps)
        # n_weight = n_reps / np.sum(n_reps)

        pd.set_option('precision', prec)
        RMSE = np.sqrt(np.sum([res["RMSE"] ** 2 * n for n, res in zip(n_reps, results)], 0) / total_rep)
        
        bias = arr([np.sum([np.mean(res[est]["error"]) * n for n, res in zip(n_reps, results)], 0) / total_rep for est in self.names
                   ])
        # should deal with this line
        est_std = arr([np.sum([np.std(res[est]["error"]) * n for n, res in zip(n_reps, results)], 0) / total_rep for est in self.names
                      ])

        MAE = np.sum([res["MAE"] * n for n, res in zip(n_reps, results)], 0) / total_rep

        std = np.sum([res["std"] * n for n, res in zip(n_reps, results)], 0) / total_rep
        freq = np.stack([res["freq"].T * n for n, res in zip(n_reps, results)], axis = 0)
        freq = np.sum(freq, axis = 0)  / np.sum(n_reps)

        
        res_array = np.hstack([RMSE[:, np.newaxis]
                  , MAE[:, np.newaxis]
                  , bias[:, np.newaxis]
                  , est_std[:, np.newaxis]
                  , std[:, np.newaxis] # width
                  , freq])
        res = DF(res_array
              , columns = ["RMSE", "MAE", "bias", "std", "ave_std", "freq: 0.95", "freq: 0.9"] #  "freq: 0.99",
              , index = self.names)
        display(res)

        RMSE_Q = np.sqrt(np.sum([np.mean((arr(res["raw_Q"]) - arr(res["V_true"])) ** 2) * n for n, res in zip(n_reps, results)]) / total_rep)
        MAE_Q = np.sum([np.sum(np.abs(arr(res["raw_Q"]) - arr(res["V_true"]))) for n, res in zip(n_reps, results)])  / total_rep
        #np.mean(np.abs(arr(self.raw_Q) - arr(self.V_true)))

        print("Q: RMSE = {:.2f}, MAE = {:.2f}".format(RMSE_Q, MAE_Q))

        printR("rep = {}".format(total_rep))
        return res_array

    def print_one_seed(self, V_true, are = None, prec = 3):
        from IPython.display import display
        raw_Qs, DR_V, TR_V, QR_V = are.raw_Qs, are.DR_V, are.TR_V, are.QR_V

        printR("true value: {:.2f} ".format(V_true))
        printR("raw Q-value: {:.2f}".format(np.mean(raw_Qs)))
        printR("raw IS: {:.2f} with std = {:.2f} ".format(are.IS_V["V"], are.IS_V["sigma"]))

        pd.set_option('precision', prec)
        printR("DR: est = {:.2f}, sigma = {:.2f}".format(DR_V["V"], DR_V["sigma"]))
        display(DF(DR_V["CIs"], index = ["0.05", "0.1"]))
        printR("TR: est = {:.2f}, sigma = {:.2f}".format(TR_V["V"], TR_V["sigma"]))
        display(DF(TR_V["CIs"], index = ["0.05", "0.1"]))    
        printR("QR: est = {:.2f}, sigma = {:.2f}".format(QR_V["V"], QR_V["sigma"]))
        display(DF(QR_V["CIs"], index = ["0.05", "0.1"]))    

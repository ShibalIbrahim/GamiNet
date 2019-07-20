import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import time 
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from gaminet import GAMINet
from datasets import *
from benchmarks import *
from metrics import * 

repeat_num = 10
num_cores = 10
task_list = {#"simu1": {"loader":data_generator1, "task_group":"simulation", "metric":mse},
         #"simu2": {"loader":data_generator2, "task_group":"simulation", "metric":mse},
         #"wine": {"loader":load_wine, "task_group":"real_data", "metric":mse},
         #"concrete": {"loader":load_concrete, "task_group":"real_data", "metric":mse},
         #"parkinsons": {"loader":load_parkinsons, "task_group":"real_data", "metric":mse},
         #"skill_craft": {"loader":load_skill_craft, "task_group":"real_data", "metric":mse},
         #"magic": {"loader":load_magic, "task_group":"real_data", "metric":auc}, 
         #"spambase": {"loader":load_spambase, "task_group":"real_data", "metric":auc},
         #"seismic_bumps": {"loader":load_seismic_bumps, "task_group":"real_data", "metric":auc},
         "credit_default":{"loader":load_credit_default, "task_group":"real_data", "metric":auc}}


def batch_parallel(method, task_group, folder, data_loader, metric, rand_seed):
    if task_group == "simulation":
        train_x, test_x, train_y, test_y, task_type, meta_info = data_loader(10000, 10000, noise_sigma=1, rand_seed=rand_seed)
    elif task_group == "real_data":
        train_x, test_x, train_y, test_y, task_type, meta_info = data_loader(path="./data/", test_ratio=0.2, rand_seed=rand_seed)

    if method == "GAMINet":
        tf.random.set_seed(rand_seed)
        model = GAMINet(input_num=train_x.shape[1], meta_info=meta_info, inter_num=10, inter_subnet_arch=[20, 10],
                       subnet_arch=[40, 20], task_type=task_type, 
                       activation_func=tf.tanh, batch_size=min(1000, int(round(0.2*train_x.shape[0]))), lr_bp=0.001, beta_threshold=0.05,
                       init_training_epochs=2000, interact_training_epochs=2000, tuning_epochs=500,
                       l1_subnet=0.01, l1_inter=0.01, verbose=False, val_ratio=0.2, early_stop_thres=50)
        model.fit(train_x, train_y)
        model.visualize(folder + "/gaminet/", "R_" + str(rand_seed + 1).zfill(2))
        pred_train = model.predict(train_x)
        pred_test = model.predict(test_x)

    elif method == "EBM":
        pred_train, pred_test, ebm_clf = ebm(train_x, train_y, test_x, task_type=task_type, meta_info=meta_info, rand_seed=rand_seed)
        ebm_visualize(ebm_clf, meta_info, folder + "/ebm/", "R_" + str(rand_seed + 1).zfill(2), cols_per_row=3, save_eps=False)
    elif method == "Rulefit":
        pred_train, pred_test = rulefit(train_x, train_y, test_x, task_type=task_type, meta_info=meta_info, rand_seed=rand_seed)
    elif method == "Hiernet":
        pred_train, pred_test = hiernet(train_x, train_y, test_x, task_type=task_type, meta_info=meta_info)
    elif method == "MLP":
        pred_train, pred_test = mlp(train_x, train_y, test_x, task_type=task_type, meta_info=meta_info, rand_seed=rand_seed)
    elif method == "RF":
        pred_train, pred_test = rf(train_x, train_y, test_x, task_type=task_type, meta_info=meta_info, rand_seed=rand_seed)
    
    res_stat = pd.DataFrame(np.array([np.round(metric(train_y, pred_train),5), 
                     np.round(metric(test_y, pred_test),5)]).reshape([1,-1]), 
                     columns=['train_metric', "test_metric"])
    return res_stat


for task_name, item in task_list.items():

    print(task_name)
    folder = "./results/" + task_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    task_group = item['task_group']
    data_loader = item['loader']
    metric = item['metric']
    
    start = time.time()
    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("GAMINet", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    gaminet_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("GAMINet Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("EBM", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    ebm_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("EBM Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("Rulefit", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    rulefit_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("Rulefit Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("Hiernet", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    hiernet_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("HierNet Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("MLP", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    mlp_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("MLP Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = Parallel(n_jobs=num_cores)(delayed(batch_parallel)("RF", task_group, folder, data_loader, metric, rand_seed) for rand_seed in range(repeat_num))
    rf_stat = pd.concat(stat).loc[:,['train_metric', 'test_metric']].values
    print("RF Finished!", "Time Cost ", np.round(time.time() - start, 2), " Seconds!")

    stat = pd.DataFrame({"hiernet_stat_mean":hiernet_stat.mean(0), "rf_stat_mean":rf_stat.mean(0), 
          "mlp_stat_mean":mlp_stat.mean(0), "rulefit_stat_mean":rulefit_stat.mean(0),
         "ebm_stat_mean":ebm_stat.mean(0), "gaminet_stat_mean":gaminet_stat.mean(0)}, index=["train_metric", "test_metric"]).T

    stat.round(5).to_csv(folder + task_name + ".csv")

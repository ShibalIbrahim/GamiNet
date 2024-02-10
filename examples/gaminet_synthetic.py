from __future__ import division
import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.

from copy import deepcopy

import os
import gc
# from ipywidgets import *
import numpy as np
import pandas as pd
from scipy.special import comb
import argparse
import pathlib
from sklearn.metrics import mean_squared_error
from IPython.display import display
from scipy import stats

from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.model_selection import KFold

import optuna
from optuna.samplers import RandomSampler
import timeit
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow as tf

import sys
sys.path.insert(0, os.path.abspath(os.getcwd()).split('examples')[0])
from gaminet import GAMINet

                
parser = argparse.ArgumentParser(description='GamiNet on synthetic data.')

# Data Arguments
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--train_size', dest='train_size',  type=int, default=400)
parser.add_argument('--test_size', dest='test_size',  type=float, default=10000)
parser.add_argument('--dataset', dest='dataset',  type=str, default='synthetic')
parser.add_argument('--correlation', dest='correlation',  type=float, default=0.0)
parser.add_argument('--dist', dest='dist',  type=str, default='normal')

# Logging Arguments
parser.add_argument('--path', dest='path',  type=str, default='/home/gridsan/shibal')
parser.add_argument('--version', dest='version',  type=int, default=1)


args = parser.parse_args()
                
      
np.random.seed(args.seed)

if args.dataset == 'synthetic':
    p = 10
    Xtrain = np.random.random((args.train_size, p))
    Xtest = np.random.random((args.test_size, p))

    def g0(t):
        return t

    def g1(t):
        return (2*t - 1)**2

    def g2(t):
        return np.sin(2*np.pi*t)/(2-np.sin(2*np.pi*t))

    def g3(t):
        return 0.1*np.sin(2*np.pi*t)+0.2*np.cos(2*np.pi*t)+0.3*(np.sin(2*np.pi*t)**2)+0.4*(np.cos(2*np.pi*t)**3)+0.5*(np.sin(2*np.pi*t)**3)                          
    
    def get_f(x):
        f = g0(x[:,0])+g1(x[:,1])+g2(x[:,2])+g3(x[:,3])+g0(x[:,2]*x[:,3])+g1(0.5*(x[:,0]+x[:,2]))+g2(x[:,0]*x[:,1])
        return f
    ftrain = get_f(Xtrain)
    ftest = get_f(Xtest)

    if args.dist == 'normal':
        errortrain = np.random.normal(loc=0, scale=0.2546, size=ftrain.shape)
        errortest = np.random.normal(loc=0, scale=0.2546, size=ftest.shape)
    elif args.dist == 'skewed':
        errortrain = stats.lognorm(s=0.2546, loc=-1.0).rvs(size=ftrain.shape)
        errortest = stats.lognorm(s=0.2546, loc=-1.0).rvs(size=ftest.shape)
    elif args.dist == 'heteroskedastic':
        errortrain = np.random.normal(loc=0, scale=2*0.2546*g1(Xtrain[:,4]))
        errortest = np.random.normal(loc=0, scale=2*0.2546*g1(Xtest[:,4]))
    else:
        raise ValueError(f"Error distribution {args.dist} is not supported")

    ytrain = ftrain+errortrain
    ytest = ftest+errortest
    ytrain = ytrain.reshape(-1,1)
    ytest = ytest.reshape(-1,1)  
    num_of_folds = 5
    main_support_true = np.array([1,1,1,1,0,0,0,0,0,0])
    interaction_terms_all = []
    for m in range(0, p):
        for n in range(0, p):
            if m!=n and m<n:
                interaction_terms_all.append((m, n))
    interaction_terms_all = np.array(interaction_terms_all)
    interaction_support_true = np.zeros((len(interaction_terms_all)))
    for term in np.array([[0,1],[0,2],[2,3]]):
        interaction_support_true[(term.reshape(1,-1)==interaction_terms_all).all(axis=1)] = 1
        
    batch_size = 32 # default batch size of 200 gave worse results.
        
elif args.dataset == 'large-synthetic-correlated':
    p = 500
    k = 10
    correlated = True
    sigma = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            sigma[i,j] = 0.5**(abs(i-j))

    np.random.seed(args.seed)
    
    Xtrain = np.random.multivariate_normal(np.zeros(p), sigma, (int)(args.train_size))
    Xval = np.random.multivariate_normal(np.zeros(p), sigma, (int)(0.1*args.train_size))
    Xtest = np.random.multivariate_normal(np.zeros(p), sigma, args.test_size)
    feature_support_truth = np.zeros(p)
    true_support = np.arange((int)(p/(2*k)),p,(int)(p/k))
    print("True Support: ", true_support)
    feature_support_truth[true_support] = 1

    def g0(t):
        return 0.5*t
    
    def g1(t):
        return 1.25*np.sin(t)
    
    def g2(t):
        return 0.3*np.exp(t)

    def g3(t):
        return 0.5*(t**2)

    def g4(t):
        return 0.9*np.cos(t)

    def g5(t):
        return 1/(1+np.exp(-t)) # not sure about variance

    
    
    def get_f(x):
        f = g0(x[:,true_support[0]]) + \
        g1(x[:,true_support[1]]) + \
        g2(x[:,true_support[2]]) + \
        g3(x[:,true_support[3]]) + \
        g4(x[:,true_support[4]]) + \
        g5(x[:,true_support[5]]) + \
        g0(x[:,true_support[6]]) + \
        g1(x[:,true_support[7]]) + \
        g2(x[:,true_support[8]]) + \
        g3(x[:,true_support[9]]) + \
        g0(x[:,true_support[0]])*g1(x[:,true_support[1]]) +\
        g0(x[:,true_support[0]])*g2(x[:,true_support[2]]) +\
        g3(0.5*(x[:,true_support[2]]+x[:,true_support[3]])) +\
        g3(x[:,true_support[3]])*g4(x[:,true_support[4]]) +\
        g3(x[:,true_support[3]])*g5(x[:,true_support[5]]) +\
        g4(x[:,true_support[6]]*x[:,true_support[7]]) +\
        g5(x[:,true_support[8]]*x[:,true_support[9]]) +\
        g3(x[:,true_support[5]]*x[:,true_support[9]]) 
        return f
    ftrain = get_f(Xtrain)
    fval = get_f(Xval)
    ftest = get_f(Xtest)

    errortrain = np.random.normal(loc=0, scale=0.25, size=ftrain.shape)
    errorval = np.random.normal(loc=0, scale=0.25, size=fval.shape)
    errortest = np.random.normal(loc=0, scale=0.25, size=ftest.shape)

    ytrain = ftrain+errortrain
    yval = fval+errorval
    ytest = ftest+errortest
    ytrain = ytrain.reshape(-1,1)
    yval = yval.reshape(-1,1)
    ytest = ytest.reshape(-1,1)   
    num_of_folds = 1
    main_support_true = np.zeros(p)
    main_support_true[true_support] = 1
    interaction_terms_all = []
    for m in range(0, p):
        for n in range(0, p):
            if m!=n and m<n:
                interaction_terms_all.append((m, n))
    interaction_terms_all = np.array(interaction_terms_all)
    interaction_support_true = np.zeros((len(interaction_terms_all)))
    for term in np.array(
        [
            [true_support[0],true_support[1]], [true_support[0],true_support[2]], [true_support[2],true_support[3]],
            [true_support[3],true_support[4]], [true_support[3],true_support[5]],
            [true_support[6],true_support[7]], [true_support[8],true_support[9]], [true_support[5],true_support[9]],
        ]
    ):
        interaction_support_true[(term.reshape(1,-1)==interaction_terms_all).all(axis=1)] = 1
    batch_size = 200 # default batch size


def identity_func(x):
    return np.array(x)
y_preprocessor = FunctionTransformer(lambda x: np.array(x)) # acts as identity
y_scaler = y_preprocessor

save_directory = os.path.join(args.path, "elaan/baselines/GamiNet/examples/results", args.dataset, args.dist, args.correlation, "N_train_{}".format(args.train_size), "seed{}".format(args.seed)) 

column_names = np.arange(Xtrain.shape[1])
logging = True
version = args.version


eval_criteria = 'mse'
_, p = Xtrain.shape


path = os.path.join(
    save_directory,
    'GamiNet',
    'v{}'.format(version),
)
os.makedirs(path, exist_ok=True)



def objective(trial, X, Y, Xval, Yval, meta_info, dataset):
    
    if dataset=='synthetic':
        num_of_interactions = trial.suggest_categorical('num_of_interactions', [5,15,25,35,45])
    elif dataset in ['large-synthetic', 'large-synthetic-correlated']:
        num_of_interactions = trial.suggest_categorical('num_of_interactions', [5,50])
    
    num_epochs = 500
    patience = 50
    learning_rate = 0.0001
    model = GAMINet(
        meta_info=meta_info,
        interact_num=num_of_interactions, 
        interact_arch=[40] * 5,
        subnet_arch=[40] * 5, 
        batch_size=batch_size,
        task_type='Regression',
        activation_func=tf.nn.relu, 
        main_effect_epochs=num_epochs,
        interaction_epochs=num_epochs,
        tuning_epochs=num_epochs, 
        lr_bp=[learning_rate, learning_rate, learning_rate],
        early_stop_thres=[patience, patience, patience],
        heredity=True,
        loss_threshold=0.0,
        reg_clarity=1,
        verbose=True,
        val_ratio=0.1, # doesn't matter as val data is explicitly passed below
    )

    model.fit(X, Y, valid_x=Xval, valid_y=Yval)
    mse_train = mean_squared_error(Y, model.predict(X))
    mse_valid = mean_squared_error(Yval, model.predict(Xval))
    print("Train: MSE:", mse_train)
    print("Val: MSE:", mse_valid)

    trial.set_user_attr("mse_train", mse_train)
    trial.set_user_attr("mse_valid", mse_valid)
    
    l_features_main = np.array([x[1:] for x in model.feature_list_], dtype=int)
    l_features_main = l_features_main[model.active_main_effect_index]
    print(l_features_main)
    l_features_inter = np.array(model.interaction_list)
    l_features_inter = l_features_inter[model.active_interaction_index]
    print(l_features_inter)
    n_z_i = len(l_features_main)
    n_z_ij = len(np.unique(l_features_inter.flatten()))
    n_features_used = len(np.unique(np.concatenate([l_features_main,l_features_inter.flatten()]))) 
    
    trial.set_user_attr("main_effects", l_features_main)
    trial.set_user_attr("interaction_effects", l_features_inter)
    trial.set_user_attr("num_main_effects", n_z_i)
    trial.set_user_attr("num_interaction_effects", n_z_ij)
    trial.set_user_attr("num_features", n_features_used)
    
    
    return mse_valid

if args.dataset=='synthetic':
    kf = KFold(n_splits=num_of_folds, random_state=None)
    kf.get_n_splits(Xtrain)

    df_studies = []

    for fold, (train_index, val_index) in enumerate(kf.split(Xtrain)):
        print("===================FOLD: {} ================".format(fold))
    #     print("TRAIN:", train_index, "VAL:", val_index)
        X_train, X_val = Xtrain[train_index], Xtrain[val_index]
        y_train, y_val = ytrain[train_index], ytrain[val_index]

        path_fold = os.path.join(path,'fold{}'.format(fold))
        os.makedirs(path_fold, exist_ok=True)


        task_type = 'Regression'
        meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(X_train.shape[1])}
        meta_info.update({'Y':{'type':'target'}})         

        for i, (key, item) in enumerate(meta_info.items()):
            if item['type'] == 'target':
                def identity_func(x):
                    return np.array(x)
                sy = FunctionTransformer(lambda x: np.array(x)) # acts as identity
        #             sy = MinMaxScaler((0, 1))
                y_train = sy.fit_transform(y_train)
                y_val = sy.transform(y_val)

                meta_info[key]['scaler'] = sy
            else:
                sx = MinMaxScaler((0, 1))
                X_train[:,[i]] = sx.fit_transform(X_train[:,[i]])
                X_val[:,[i]] = sx.transform(X_val[:,[i]])
                meta_info[key]['scaler'] = sx

        ntrials = 5
        start = timeit.default_timer()

        study = optuna.create_study(sampler=RandomSampler(seed=0), direction='minimize')
        objective_wrapper = lambda trial: objective(trial, X_train, y_train, X_val, y_val, meta_info, args.dataset)
        study.optimize(objective_wrapper, n_trials=ntrials)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        stop = timeit.default_timer()
        hours, rem = divmod(stop-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} trials".format(int(hours), int(minutes), seconds, ntrials))     

        df_study = study.trials_dataframe()
        df_studies.append(df_study)
        
    ###### Read csv files per fold to find optimal hyperparameters        
    for fold in range(num_of_folds):
        print(fold)
        df_temp = df_studies[fold].set_index(['params_num_of_interactions'])[['value']]
        df_temp.columns = ['val-{}'.format(fold)]
        if fold==0:
            df = df_temp.copy()
        else:
            df = df.join(df_temp, how='outer')    
        display(df)
    dfr = df.reset_index()
    dfr = dfr.sort_values(by=['params_num_of_interactions'], ascending=False).set_index(['params_num_of_interactions'])
    dfr = dfr.mean(axis=1)
    dfr = dfr[dfr==dfr.min()].reset_index()        
    display(dfr)
    with open(path+'/Results.txt', "a") as f:
        f.write('\n CV MSE: {}\n'.format(dfr.values))
    num_of_interactions_opt = (int)(dfr['params_num_of_interactions'].values[0])
            
elif args.dataset=='large-synthetic-correlated':
#     X_train, X_val = deepcopy(Xtrain), deepcopy(Xval)
#     y_train, y_val = deepcopy(ytrain), deepcopy(yval)

#     path_fold = os.path.join(path,'fold{}'.format(0))
#     os.makedirs(path_fold, exist_ok=True)


#     task_type = 'Regression'
#     meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(X_train.shape[1])}
#     meta_info.update({'Y':{'type':'target'}})         

#     for i, (key, item) in enumerate(meta_info.items()):
#         if item['type'] == 'target':
#             def identity_func(x):
#                 return np.array(x)
#             sy = FunctionTransformer(lambda x: np.array(x)) # acts as identity
#     #             sy = MinMaxScaler((0, 1))
#             y_train = sy.fit_transform(y_train)
#             y_val = sy.transform(y_val)

#             meta_info[key]['scaler'] = sy
#         else:
#             sx = MinMaxScaler((0, 1))
#             X_train[:,[i]] = sx.fit_transform(X_train[:,[i]])
#             X_val[:,[i]] = sx.transform(X_val[:,[i]])
#             meta_info[key]['scaler'] = sx

#     ntrials = 2
#     start = timeit.default_timer()

#     study = optuna.create_study(sampler=RandomSampler(seed=0), direction='minimize')
#     objective_wrapper = lambda trial: objective(trial, X_train, y_train, X_val, y_val, meta_info, args.dataset)
#     study.optimize(objective_wrapper, n_trials=ntrials)

#     print("Number of finished trials: {}".format(len(study.trials)))

#     print("Best trial:")
#     best_trial = study.best_trial

#     print("  Value: {}".format(best_trial.value))

#     print("  Params: ")
#     for key, value in best_trial.params.items():
#         print("    {}: {}".format(key, value))

#     stop = timeit.default_timer()
#     hours, rem = divmod(stop-start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} trials".format(int(hours), int(minutes), seconds, ntrials))     

#     df_study = study.trials_dataframe()
#     df = df_study.set_index(['params_num_of_interactions'])[['value']]
#     df.columns = ['val-{}'.format(0)]    
    

#     dfr = df.reset_index()
#     dfr = dfr.sort_values(by=['params_num_of_interactions'], ascending=False).set_index(['params_num_of_interactions'])
#     dfr = dfr.mean(axis=1)
#     dfr = dfr[dfr==dfr.min()].reset_index()        
#     display(dfr)
#     with open(path+'/Results.txt', "a") as f:
#         f.write('\n CV MSE: {}\n'.format(dfr.values))
#     num_of_interactions_opt = (int)(dfr['params_num_of_interactions'].values[0])
#     print("num_of_interactions_opt:", num_of_interactions_opt)
    num_of_interactions_opt = 50
    print("num_of_interactions_opt:", num_of_interactions_opt)

gc.collect()

task_type = 'Regression'
meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(Xtrain.shape[1])}
meta_info.update({'Y':{'type':'target'}})         

for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        def identity_func(x):
            return np.array(x)
        sy = FunctionTransformer(lambda x: np.array(x)) # acts as identity
#             sy = MinMaxScaler((0, 1))
        ytrain = sy.fit_transform(ytrain)
        ytest = sy.transform(ytest)
        if args.dataset=='large-synthetic-correlated':
            yval = sy.transform(yval)

        meta_info[key]['scaler'] = sy
    else:
        sx = MinMaxScaler((0, 1))
        sx.fit([[0], [1]])
        Xtrain[:,[i]] = sx.transform(Xtrain[:,[i]])
        Xtest[:,[i]] = sx.transform(Xtest[:,[i]])
        if args.dataset=='large-synthetic-correlated':
            Xval[:,[i]] = sx.transform(Xval[:,[i]])
        meta_info[key]['scaler'] = sx

###### Refit for optimal smoothness on train + val
num_epochs = 500
patience = 50
learning_rate = 0.0001
model = GAMINet(
    meta_info=meta_info,
    interact_num=num_of_interactions_opt, 
    interact_arch=[40] * 5,
    subnet_arch=[40] * 5, 
    batch_size=batch_size,
    task_type='Regression',
    activation_func=tf.nn.relu, 
    main_effect_epochs=num_epochs,
    interaction_epochs=num_epochs,
    tuning_epochs=num_epochs, 
    lr_bp=[learning_rate, learning_rate, learning_rate],
    early_stop_thres=[patience, patience, patience],
    heredity=True,
    loss_threshold=0.0,
    reg_clarity=1,
    verbose=True,
    val_ratio=0.1,
)

if args.dataset=='synthetic':
    model.fit(Xtrain, ytrain)
elif args.dataset=='large-synthetic-correlated':
    model.fit(Xtrain, ytrain, valid_x=Xval, valid_y=yval)
ftest_predict = model.predict(Xtest)
true_error = mean_squared_error(ftest, ftest_predict)


# print(model.feature_names)

# l_features_main = l_features_main[model.active_main_effect_index]
# l_features_inter = np.array(model.interaction_list)
# l_features_inter = l_features_inter[model.active_interaction_index]
# n_z_i = len(l_features_main)
# n_z_ij = len(np.unique(l_features_inter.flatten()))
# n_features_used = len(np.unique(np.concatenate([l_features_main,l_features_inter.flatten()]))) 

main_effects = np.array([x[1:] for x in model.feature_list_], dtype=int)
main_effects = main_effects[model.active_main_effect_index]
main_effects -= 1
interaction_effects = np.array(model.interaction_list)
interaction_effects = interaction_effects[model.active_interaction_index]

# Compute FPR and FNR for main effects
main_support_recovered = np.zeros_like(main_support_true)
main_support_recovered[main_effects] = 1
print("main_support_true:", main_support_true)
print("main_support_recovered:", main_support_recovered)

tpr_main = recall_score(main_support_true, main_support_recovered)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnr_main = recall_score(main_support_true, main_support_recovered, pos_label=0) 
fpr_main = 1 - tnr_main
fnr_main = 1 - tpr_main   
f1_main = f1_score(main_support_true, main_support_recovered)

# Compute FPR and FNR for interactions
interaction_support_recovered = np.zeros((len(interaction_terms_all)))
if len(interaction_effects)>0:
    for term in interaction_effects:
        interaction_support_recovered[(term.reshape(1,-1)==interaction_terms_all).all(axis=1)] = 1

print("interaction_support_true:", interaction_support_true)
print("interaction_support_recovered:", interaction_support_recovered)
   
tpr_interaction = recall_score(interaction_support_true, interaction_support_recovered)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnr_interaction = recall_score(interaction_support_true, interaction_support_recovered, pos_label=0) 
fpr_interaction = 1 - tnr_interaction
fnr_interaction = 1 - tpr_interaction 
f1_interaction = f1_score(interaction_support_true, interaction_support_recovered)
        
with open(path+'/Results.txt', "a") as f:
    f.write('\n True Test MSE: {}\n'.format(true_error))
    f.write('FPR (main): {}\n'.format(fpr_main))
    f.write('FNR (main): {}\n'.format(fnr_main))
    f.write('F1 (main): {}\n'.format(f1_main))
    f.write('FPR (interactions): {}\n'.format(fpr_interaction))
    f.write('FNR (interactions): {}\n'.format(fnr_interaction))
    f.write('F1 (interactions): {}\n'.format(f1_interaction))
    f.write('Main-effects: {}\n'.format(main_effects))
    f.write('Interaction-effects: {}\n'.format(interaction_effects))

with open(path+'/support_set.npy', 'wb') as f:
    np.save(f, main_effects)
    np.save(f, interaction_effects)

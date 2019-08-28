import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

from interpret.glassbox import ExplainableBoostingRegressor
from interpret.glassbox import ExplainableBoostingClassifier 

import rpy2
import rpy2.robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
pandas2ri.activate()
ro.r('sink("/dev/null")')
ro.r['options'](warn=-1)


def preprocessing(train_x, test_x, meta_info):
    new_train = []
    new_test = []
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            continue
        if item['type'] == "categorical":
            if len(item['values']) > 2:
                ohe = OneHotEncoder(sparse=False, drop='first', categories='auto')
                temp = ohe.fit_transform(np.vstack([train_x[:,[i]], test_x[:,[i]]]))
                new_train.append(temp[:train_x.shape[0],:])
                new_test.append(temp[train_x.shape[0]:,:])
            else:
                new_train.append(train_x[:, [i]])
                new_test.append(test_x[:, [i]])
        if item['type'] == "continuous":
            if train_x[:, [i]].std() > 0:
                new_train.append(train_x[:, [i]])
                new_test.append(test_x[:, [i]])
    new_train = np.hstack(new_train)
    new_test = np.hstack(new_test)   
    return new_train, new_test
           
    
def ebm_visualize(ebm_clf, meta_info, folder="./results/", name="demo", cols_per_row=3, main_density = 3, save_eps=False):  
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = folder + name

    idx = 0
    cols_per_row = 3
    categ_variable_list = []
    noncateg_variable_list = []
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            continue
        if item['type'] == "categorical":
            categ_variable_list.append(key)
        else:
            noncateg_variable_list.append(key)

    ebm_global = ebm_clf.explain_global()
    max_ids = len(ebm_global.feature_names)
    f = plt.figure(figsize=(6 * cols_per_row, int(max_ids * 6 / cols_per_row)))
    for indice in range(max_ids):

        data_dict = ebm_global.data(indice)
        feature_name = ebm_global.feature_names[indice]
        feature_type = ebm_global.feature_types[indice]

        if feature_type == "continuous":
            
            ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                            rowspan=main_density - 1)
            
            sx = meta_info[feature_name]["scaler"]
            subnets_inputs = np.array(data_dict['names']).reshape([-1, 1])
            subnets_inputs_real = sx.inverse_transform(subnets_inputs)
            x_tick_loc = np.linspace(int(len(subnets_inputs) * 0.1), int(len(subnets_inputs) * 0.9), 5).reshape([-1, 1]).astype(int)
                                           
            x_tick_values = subnets_inputs_real[x_tick_loc].ravel()
            if (np.max(x_tick_values) - np.min(x_tick_values)) < 0.01:
                x_tick_values = np.array([np.format_float_scientific(x_tick_values[i], precision=1) for i in range(x_tick_values.shape[0])])
            elif (np.max(x_tick_values) - np.min(x_tick_values)) < 10:
                x_tick_values = np.round(x_tick_values, 2)
            elif (np.max(x_tick_values) - np.min(x_tick_values)) < 1000:
                x_tick_values = np.round(x_tick_values).astype(int)
            else:
                x_tick_values = np.array([np.format_float_scientific(x_tick_values[i], precision=1) for i in range(x_tick_values.shape[0])])

        
            ax1.errorbar(subnets_inputs_real, data_dict['scores'], ecolor="gray",
                     yerr=np.array(data_dict['upper_bounds']) - np.array(data_dict['scores']))
            plt.xticks(subnets_inputs_real[x_tick_loc].ravel(), x_tick_values.ravel(), fontsize=10)
            ax1.set_ylabel("Score", fontsize=12)
            
            ax2 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row) + main_density - 1, idx%cols_per_row))

            xint = sx.inverse_transform(np.reshape((np.array(data_dict['density']['names'][1:])
                                       + np.array(data_dict['density']['names'][:-1]))/2, [-1,1])).reshape([-1])
            ax2.bar(xint,data_dict['density']['scores'], width=xint[1]-xint[0])
            ax2.set_ylabel("Density", fontsize=12)
            
        elif feature_type == "categorical":
            
            ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                            rowspan=main_density - 1)

            values = data_dict['scores']
            values_num = len(data_dict['scores'])
            ax1.bar(np.arange(values_num), data_dict['scores'])
            plt.xticks(np.arange(values_num), meta_info[feature_name]["values"])
            ax1.set_ylabel("Score", fontsize=12)
            
            ax2 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row) + main_density - 1, idx%cols_per_row))

            unique, counts = np.unique(self.tr_x[:, 
                              self.categ_index_list[indice - self.numerical_input_num]], return_counts=True)
            ax2.bar(np.arange(len(meta_info[feature_name]['values'])), counts)
            plt.xticks(np.arange(len(meta_info[feature_name]['values'])), 
                    meta_info[feature_name]['values'])
            ax2.set_ylabel("Density", fontsize=12)

        elif feature_type == "pairwise":
                        
            ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                            rowspan=main_density)

            response = data_dict['scores'].T[::-1]

            feature_name1 = feature_name.split(" x ")[0]
            feature_name2 = feature_name.split(" x ")[1]
            depth1, depth2 = data_dict['scores'].shape
            if (feature_name1 in categ_variable_list) & (feature_name2 not in categ_variable_list):
                
                sx2 = meta_info[feature_name.split(" x ")[1]]["scaler"]
                x1_tick_loc = np.arange(depth1)
                x2_tick_loc = np.round(np.linspace(1, data_dict['scores'].shape[1] - 1, 6)).astype(int)
                x1_real_values = np.array(meta_info[feature_name1]["values"])[x1_tick_loc].tolist()
                x2_real_values = sx2.inverse_transform(np.array(data_dict['right_names'])[x2_tick_loc].reshape([-1, 1])).ravel()[::-1]

            elif (feature_name1 not in categ_variable_list) & (feature_name2 in categ_variable_list):
                
                sx1 = meta_info[feature_name.split(" x ")[0]]["scaler"]
                x1_tick_loc = np.round(np.linspace(1, data_dict['scores'].shape[0] - 1, 6)).astype(int)
                x2_tick_loc = np.arange(depth2) 
                x1_real_values = sx1.inverse_transform(np.array(data_dict['left_names'])[x1_tick_loc].reshape([-1, 1])).ravel()
                x2_real_values = np.array(meta_info[feature_name2]["values"])[x2_tick_loc].tolist()[::-1]

            elif (feature_name1 in categ_variable_list) & (feature_name2 in categ_variable_list):

                x1_tick_loc = np.arange(depth1)
                x2_tick_loc = np.arange(depth2)
                x1_real_values = np.array(meta_info[feature_name1]["values"])[x1_tick_loc].tolist()
                x2_real_values = np.array(meta_info[feature_name2]["values"])[x2_tick_loc].tolist()[::-1]

            else:
                sx1 = meta_info[feature_name1]["scaler"]
                sx2 = meta_info[feature_name2]["scaler"]

                x1_tick_loc = np.round(np.linspace(1, data_dict['scores'].shape[0] - 1, 6)).astype(int)
                x2_tick_loc = np.round(np.linspace(1, data_dict['scores'].shape[1] - 1, 6)).astype(int)
                x1_real_values = sx1.inverse_transform(np.array(data_dict['left_names'])[x1_tick_loc].reshape([-1, 1])).ravel()
                x2_real_values = sx2.inverse_transform(np.array(data_dict['right_names'])[x2_tick_loc].reshape([-1, 1])).ravel()[::-1]
                
            if feature_name1 not in categ_variable_list:

                if (np.max(x1_real_values) - np.min(x1_real_values)) < 0.01:
                    x1_real_values = np.array([np.format_float_scientific(x1_real_values[i], 
                                      precision=1) for i in range(x1_real_values.shape[0])])
                elif (np.max(x1_real_values) - np.min(x1_real_values)) < 10:
                    x1_real_values = np.round(x1_real_values, 2)
                elif (np.max(x1_real_values) - np.min(x1_real_values)) < 1000:
                    x1_real_values = np.round(x1_real_values).astype(int)
                else:
                    x1_real_values = np.array([np.format_float_scientific(x1_real_values[i],
                                      precision=1) for i in range(x1_real_values.shape[0])])

            if feature_name2 not in categ_variable_list:

                if (np.max(x2_real_values) - np.min(x2_real_values)) < 0.01:
                    x2_real_values = np.array([np.format_float_scientific(x2_real_values[i],
                                      precision=1) for i in range(x2_real_values.shape[0])])
                elif (np.max(x2_real_values) - np.min(x2_real_values)) < 10:
                    x2_real_values = np.round(x2_real_values, 2)
                elif (np.max(x2_real_values) - np.min(x2_real_values)) < 1000:
                    x2_real_values = np.round(x2_real_values).astype(int)
                else:
                    x2_real_values = np.array([np.format_float_scientific(x2_real_values[i],
                                      precision=1) for i in range(x2_real_values.shape[0])])

            cf = ax1.imshow(response, interpolation='nearest', aspect='auto')
            plt.xticks(x1_tick_loc, x1_real_values, fontsize=10)
            plt.yticks(x2_tick_loc, x2_real_values, fontsize=10)
            response_precision = max(int(- np.log10(np.max(response) - np.min(response))) + 2, 0)
            f.colorbar(cf, ax=ax1, format='%0.' + str(response_precision) + 'f', orientation='horizontal')

        ax1.set_title(feature_name, fontsize=12)
        idx = idx + 1
    f.tight_layout()
    if max_ids > 0:
        f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
        if save_eps:
            f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)


def rf(train_x, train_y, test_x, meta_info, task_type="Regression", rand_seed=0):

    train_x, test_x = preprocessing(train_x, test_x, meta_info)
    if task_type == "Regression":
        base = RandomForestRegressor(n_estimators=100, random_state=rand_seed)
        grid = GridSearchCV(base, param_grid={'max_depth': (3, 4, 5, 6, 7, 8)},
                            scoring={'mse' : make_scorer(mean_squared_error, greater_is_better=False)},
                            cv=3, refit='mse', error_score=-np.inf)
        grid.fit(train_x, train_y.ravel())
        model = grid.best_estimator_
        pred_train = model.predict(train_x).reshape([-1, 1])
        pred_test = model.predict(test_x).reshape([-1, 1])

    elif task_type == "Classification":
        base = RandomForestClassifier(n_estimators=100, random_state=rand_seed)
        grid = GridSearchCV(base, param_grid={'max_depth': (3, 4, 5, 6, 7, 8)},
                            scoring={'auc' : make_scorer(roc_auc_score)},
                            cv=3, refit='auc')
        grid.fit(train_x, train_y.ravel())
        model = grid.best_estimator_
        pred_train = model.predict_proba(train_x)[:, 1:]
        pred_test = model.predict_proba(test_x)[:, 1:]
    return pred_train, pred_test


def mlp(train_x, train_y, test_x, meta_info, task_type="Regression", epoches=10000, early_stop=100, rand_seed=0):
    
    datanum = train_x.shape[0]
    train_x, test_x = preprocessing(train_x, test_x, meta_info)

    if task_type == "Regression":
        model = MLPRegressor(hidden_layer_sizes=[100, 60], max_iter=epoches, alpha=0.0, batch_size=min(1000, int(np.floor(datanum * 0.20))), \
                      activation='tanh', tol=0, early_stopping=True,
                      random_state=rand_seed, validation_fraction=0.2, n_iter_no_change=early_stop)
        model.fit(train_x, train_y.ravel())
        pred_train = model.predict(train_x).reshape([-1, 1])
        pred_test = model.predict(test_x).reshape([-1, 1])
    elif task_type == "Classification":
        model = MLPClassifier(hidden_layer_sizes=[100, 60], max_iter=epoches, alpha=0.0, batch_size=min(1000, int(np.floor(datanum * 0.20))), \
                       activation='tanh', tol=0, early_stopping=True,
                       random_state=rand_seed, validation_fraction=0.2, n_iter_no_change=early_stop)
        model.fit(train_x, train_y.ravel())
        pred_train = model.predict_proba(train_x)[:, 1:]
        pred_test = model.predict_proba(test_x)[:, 1:]

    return pred_train, pred_test


# install
# utils = importr('devtools')
# utils.install_version("hierNet", version = "1.7", repos = "http://cran.r-project.org")
def hiernet(train_x, train_y, test_x, meta_info, task_type="Regression"):

    hn = importr("hierNet")
    train_x, test_x = preprocessing(train_x, test_x, meta_info)
    if task_type == "Regression":
        fit=hn.hierNet_path(ro.r.matrix(train_x, nrow=train_x.shape[0], ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()))
        fitcv=hn.hierNet_cv(fit, ro.r.matrix(train_x, nrow=train_x.shape[0],
                     ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()), nfolds=3, trace=0)
        clf=hn.hierNet(ro.r.matrix(train_x, nrow=train_x.shape[0],
                   ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()), lam=fitcv[4][0], trace=0)
        pred_train = np.array(hn.predict_hierNet(clf, ro.r.matrix(train_x, nrow=train_x.shape[0], ncol=train_x.shape[1]))).reshape([-1,1])
        pred_test = np.array(hn.predict_hierNet(clf, ro.r.matrix(test_x, nrow=test_x.shape[0], ncol=test_x.shape[1]))).reshape([-1,1])
    elif task_type == "Classification":
        fit=hn.hierNet_logistic_path(ro.r.matrix(train_x, nrow=train_x.shape[0], ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()))
        fitcv=hn.hierNet_cv(fit, ro.r.matrix(train_x, nrow=train_x.shape[0],
                                 ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()), nfolds=3, trace=0)
        clf=hn.hierNet_logistic(ro.r.matrix(train_x, nrow=train_x.shape[0],
                                 ncol=train_x.shape[1]), ro.vectors.FloatVector(train_y.ravel()), lam=fitcv[4][0], trace=0)
        pred_train = np.array(hn.predict_hierNet_logistic(clf, ro.r.matrix(train_x, 
                                           nrow=train_x.shape[0], ncol=train_x.shape[1])))[0,:].reshape([-1,1])
        pred_test = np.array(hn.predict_hierNet_logistic(clf, ro.r.matrix(test_x, 
                                          nrow=test_x.shape[0], ncol=test_x.shape[1])))[0,:].reshape([-1,1])

    return pred_train, pred_test


# utils = importr('devtools')
# utils.install_version("pre", version = "0.7.1", repos = "http://cran.r-project.org")
def rulefit(train_x, train_y, test_x, meta_info, task_type="Regression", rand_seed=0):
    
    pre = importr('pre')
    train_x, test_x = preprocessing(train_x, test_x, meta_info)
    test_x = pd.DataFrame(test_x)
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y, columns=["y"])
    r_train_x = pandas2ri.py2rpy_pandasdataframe(train_x)
    r_test_x = pandas2ri.py2rpy_pandasdataframe(test_x)

    if task_type == "Regression":
        r_train_data = pandas2ri.py2rpy_pandasdataframe(pd.concat([train_x, train_y], 1))
        fit = pre.pre(Formula('y ~ .'), data=r_train_data, family = "gaussian", nfolds=3)
    elif task_type == "Classification":
        r_train_data = pandas2ri.py2rpy_pandasdataframe(pd.concat([train_x, train_y.astype(str).astype('category')], 1))
        fit = pre.pre(Formula('y ~ .'), data=r_train_data, family = "binomial", nfolds=3)

    pred_train = pre.predict_pre(fit, newdata=r_train_x).reshape([-1,1])
    pred_test = pre.predict_pre(fit, newdata=r_test_x).reshape([-1,1])
    return pred_train, pred_test


def ebm(train_x, train_y, test_x, meta_info, task_type="Regression", rand_seed=0):

    feature_types = [item["type"] for key, item in meta_info.items()][:-1]
    feature_names = [key for key, item in meta_info.items()][:-1]
    if task_type == "Regression":
        ebm_clf = ExplainableBoostingRegressor(interactions=10, feature_names=feature_names, feature_types=feature_types, 
                                  random_state=rand_seed, holdout_size=0.2)
        ebm_clf.fit(train_x, train_y.ravel())
        pred_train = ebm_clf.predict(train_x).reshape([-1,1])
        pred_test = ebm_clf.predict(test_x).reshape([-1,1])
    elif task_type == "Classification":
        ebm_clf = ExplainableBoostingClassifier(interactions=10, feature_names=feature_names, feature_types=feature_types,
                                   random_state=rand_seed, holdout_size=0.2)
        ebm_clf.fit(train_x, train_y.ravel())
        pred_train = ebm_clf.predict_proba(train_x)[:,1]
        pred_test = ebm_clf.predict_proba(test_x)[:,1]
    return pred_train, pred_test, ebm_clf


def logr(train_x, train_y, test_x, meta_info, rand_seed=0):
    train_x, test_x = preprocessing(train_x, test_x, meta_info)

    grid = GridSearchCV(LogisticRegression(penalty='l2', random_state=rand_seed), param_grid={"C": np.logspace(-2, 2, 5)},
                        scoring={'auc' : make_scorer(roc_auc_score)},
                        cv=3, refit='auc', n_jobs=10)
    grid.fit(train_x, train_y.ravel())
    model = grid.best_estimator_
    pred_train = model.predict_proba(train_x)[:, 1:]
    pred_test = model.predict_proba(test_x)[:, 1:]
    return pred_train, pred_test, model
import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
           
    
def ebm_visualize(ebm_clf, meta_info, folder="./results/", name="demo", cols_per_row=3, save_png=False, save_eps=False):  
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = folder + name

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
    fig = plt.figure(figsize=(6 * cols_per_row - 2, 
                      4.6 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids/cols_per_row)), cols_per_row, wspace=0.25, hspace=0.25)
    for indice in range(max_ids):

        data_dict = ebm_global.data(indice)
        feature_name = ebm_global.feature_names[indice]
        feature_type = ebm_global.feature_types[indice]
        
        if feature_type == "continuous":

            sx = meta_info[feature_name]['scaler']
            subnets_inputs = np.array(data_dict['names']).reshape([-1, 1])
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[indice], wspace=0.1, hspace=0.1, height_ratios=[4, 1])
            ax1 = plt.Subplot(fig, inner[0]) 
            ax1.plot(sx.inverse_transform(subnets_inputs), data_dict['scores'])
            ax1.fill(np.concatenate([sx.inverse_transform(subnets_inputs), sx.inverse_transform(subnets_inputs)[::-1]]),
                             np.concatenate([np.array(data_dict['lower_bounds']), 
                           np.array(data_dict['upper_bounds'])[::-1]]), alpha=.5)
            ax1.set_ylabel("Score", fontsize=12)
            ax1.get_yaxis().set_label_coords(-0.15, 0.5)
            ax1.set_title(feature_name, fontsize=12)
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1]) 
            xint = sx.inverse_transform(((np.array(data_dict['density']['names'][1:]) 
                                          + np.array(data_dict['density']['names'][:-1]))/2).reshape([-1, 1])).reshape([-1])
            ax2.bar(xint, data_dict['density']['scores'], width=xint[1]-xint[0])
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.set_xticklabels([])
            ax2.set_ylabel("Histogram", fontsize=12)
            ax2.get_yaxis().set_label_coords(-0.15, 0.5)
            if np.max([len(str(int(ax1.get_yticks()[i]) if (ax2.get_yticks()[i] - int(ax2.get_yticks()[i])) < 0.001 
                               else ax1.get_yticks()[i].round(5))) for i in range(len(ax2.get_yticks()))]) > 5:
                ax1.yaxis.set_tick_params(rotation=20)
            if np.max([len(str(int(ax2.get_xticks()[i]) if (ax2.get_xticks()[i] - int(ax2.get_xticks()[i])) < 0.001 
                               else ax2.get_xticks()[i].round(5))) for i in range(len(ax2.get_xticks()))]) > 5:
                ax2.xaxis.set_tick_params(rotation=20)
            if np.max([len(str(int(ax2.get_yticks()[i]))) for i in range(len(ax2.get_yticks()))]) > 5:
                ax2.yaxis.set_tick_params(rotation=20)
            fig.add_subplot(ax2)

        elif feature_type == "categorical":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[indice], wspace=0.1, hspace=0.1, height_ratios=[4, 1])
            ax1 = plt.Subplot(fig, inner[0]) 
            ax1.bar(np.arange(len(data_dict['scores'])), data_dict['scores'], ecolor='black', capsize=10,
                  yerr=(np.array(data_dict['upper_bounds']) - np.array(data_dict['lower_bounds'])) / 2)
            ax1.set_ylabel("Score", fontsize=12)
            ax1.get_yaxis().set_label_coords(-0.15, 0.5)
            ax1.set_title(feature_name, fontsize=12)
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1]) 
            ax2.bar(np.arange(len(meta_info[feature_name]['values'])), data_dict['density']['scores'])
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.set_xticklabels([])
            ax2.set_xticks(np.arange(len(data_dict['scores'])))
            ax2.set_xticklabels(meta_info[feature_name]["values"])
            ax2.set_ylabel("Histogram", fontsize=12)
            ax2.get_yaxis().set_label_coords(-0.15, 0.5)
            if np.max([len(str(int(ax1.get_yticks()[i]) if (ax2.get_yticks()[i] - int(ax2.get_yticks()[i])) < 0.001 
                               else ax1.get_yticks()[i].round(5))) for i in range(len(ax2.get_yticks()))]) > 5:
                ax1.yaxis.set_tick_params(rotation=20)
            if np.max([len(str(int(ax2.get_xticks()[i]) if (ax2.get_xticks()[i] - int(ax2.get_xticks()[i])) < 0.001 
                               else ax2.get_xticks()[i].round(5))) for i in range(len(ax2.get_xticks()))]) > 5:
                ax2.xaxis.set_tick_params(rotation=20)
            if np.max([len(str(int(ax2.get_yticks()[i]))) for i in range(len(ax2.get_yticks()))]) > 5:
                ax2.yaxis.set_tick_params(rotation=20)
            fig.add_subplot(ax2)

        elif feature_type == "pairwise":

            response = data_dict['scores'].T[::-1]
            feature_name1 = feature_name.split(" x ")[0]
            feature_name2 = feature_name.split(" x ")[1]
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in categ_variable_list:
                interact_label1 = meta_info[feature_name1]['values']
                length1 = len(interact_label1)
                interact_input1 = np.array(np.arange(length1), dtype=np.float32)
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, length1 - 0.5])
            else:
                sx1 = meta_info[feature_name1]['scaler']    
                interact_input_list.append(np.array(np.linspace(-1, 1, 101), dtype=np.float32))
                interact_label1 = sx1.inverse_transform(np.array([-1, 1], dtype=np.float32).reshape([-1, 1])).ravel()
                axis_extent.extend([interact_label1.min(), interact_label1.max()])
            if feature_name2 in categ_variable_list:
                interact_label2 = meta_info[feature_name2]['values']
                length2 = len(interact_label2)
                interact_input2 = np.array(np.arange(length2), dtype=np.float32)
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, length2 - 0.5])
            else:
                sx2 = meta_info[feature_name2]['scaler']  
                interact_input_list.append(np.array(np.linspace(-1, 1, 101), dtype=np.float32))
                interact_label2 = sx2.inverse_transform(np.array([-1, 1], dtype=np.float32).reshape([-1, 1])).ravel()
                axis_extent.extend([interact_label2.min(), interact_label2.max()])

            ax = plt.Subplot(fig, outer[indice]) 
            cf = ax.imshow(response, interpolation='nearest', aspect='auto', extent=axis_extent)
            if feature_name1 in categ_variable_list:
                ax.set_xticks(interact_input1)
                ax.set_xticklabels(interact_label1)
            elif np.max([len(str(int(interact_label1[i]) if (interact_label1[i] - int(interact_label1[i])) < 0.001 
                           else interact_label1[i].round(5))) for i in range(len(interact_label1))]) > 5:
                ax.xaxis.set_tick_params(rotation=20)
            if feature_name2 in categ_variable_list:
                ax.set_yticks(interact_input2)
                ax.set_yticklabels(interact_label2)
            elif np.max([len(str(int(interact_label2[i]) if (interact_label2[i] - int(interact_label2[i])) < 0.001 
                           else interact_label2[i].round(5))) for i in range(len(interact_label2))]) > 5:
                ax.yaxis.set_tick_params(rotation=20)

            response_precision = max(int(- np.log10(np.max(response) - np.min(response))) + 2, 0)
            fig.colorbar(cf, ax=ax, format='%0.' + str(response_precision) + 'f')
            ax.set_title(feature_name, fontsize=12)
            if np.max([len(str(int(ax.get_xticks()[i]) if (ax.get_xticks()[i] - int(ax.get_xticks()[i])) < 0.001 
                               else ax.get_xticks()[i].round(5))) for i in range(len(ax.get_xticks()))]) > 5:
                ax.xaxis.set_tick_params(rotation=20)
            if np.max([len(str(int(ax.get_yticks()[i]) if (ax.get_yticks()[i] - int(ax.get_yticks()[i])) < 0.001 
                               else ax.get_yticks()[i].round(5))) for i in range(len(ax.get_yticks()))]) > 5:
                ax.yaxis.set_tick_params(rotation=20)       
            fig.add_subplot(ax)

    if max_ids > 0:
        if save_eps:
            fig.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
        if save_png:
            fig.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)


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
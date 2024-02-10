import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.
from contextlib import redirect_stdout

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import pathlib
import pandas as pd
import pickle as pk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error
import time


sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute())).split('examples')[0])
from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_trajectory
from gaminet.utils import plot_regularization

from sklearn.preprocessing import FunctionTransformer

import argparse

## Load data

def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))

def mse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.mean((pred - label)**2)

def metrics(y, ypred, y_preprocessor=None):
    """Evaluates metrics.
    
    Args:
        y:
        ypred:
        y_scaler:
        
    Returns:
        mae: mean absolute error, float scaler.
        std_err: standard error, float scaler.
    """
    if y_preprocessor is not None:
        y = y_preprocessor.inverse_transform(y)
        ypred = y_preprocessor.inverse_transform(ypred)
    
    mse = mean_squared_error(y, ypred)
    rmse = mean_squared_error(y, ypred, squared=False)
    return mse, rmse


def load_data(load_directory='./',
              filename='pdb2019trv3_us.csv',
              remove_margin_of_error_variables=False): 
    """Loads Census data, and retrieves covariates and responses.
    
    Args:
        load_directory: Data directory for loading Census file, str.
        filename: file to load, default is 'pdb2019trv3_us.csv'.
        remove_margin_of_error_variables: whether to remove margin of error variables, bool scaler.
        
    Returns:
        df_X, covariates, pandas dataframe.
        df_y, target response, pandas dataframe.
    """
    file = os.path.join(load_directory, filename)
    df = pd.read_csv(file, encoding = "ISO-8859-1")
    df = df.set_index('GIDTR')
    
    # Drop location variables
    drop_location_variables = ['State', 'State_name', 'County', 'County_name', 'Tract', 'Flag', 'AIAN_LAND']
    df = df.drop(drop_location_variables, axis=1)
    
    target_response = 'Self_Response_Rate_ACS_13_17'
    # Remove extra response variables 
    # Remove response columns 'FRST_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the first form mailed was completed and returned) and 'RPLCMNT_FRMS_CEN_2010' (Number of addresses in a 2010 Census Mailout/Mailback area where the replacement form was completed and returned)

    extra_response_variables = [
        'Census_Mail_Returns_CEN_2010',
        'Mail_Return_Rate_CEN_2010',
        'pct_Census_Mail_Returns_CEN_2010',
        'Low_Response_Score',
        'Self_Response_Rate_ACSMOE_13_17',
        'BILQ_Frms_CEN_2010',
        'FRST_FRMS_CEN_2010',
        'RPLCMNT_FRMS_CEN_2010',
        'pct_FRST_FRMS_CEN_2010',
        'pct_RPLCMNT_FRMS_CEN_2010']
    df = df.drop(extra_response_variables, axis=1)
    
    if remove_margin_of_error_variables:
        df = df[np.array([c for c in df.columns if 'MOE' not in c])]

    # Change types of covariate columns with dollar signs in their values e.g. income, housing price  
    df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].replace('[\$,]', '', regex=True).astype(np.float64)

    # Remove entries with missing predictions
    df_full = df.copy()
    df = df.dropna(subset=[target_response])

    df_y = df[[target_response]]
    df_X = df.drop([target_response], axis=1)


    return df_X, df_y, df_full

def process_data(df_X,
                 df_y,
                 val_ratio=0.1, 
                 test_ratio=0.1, 
                 seed=None,
                 standardize_response=False):
    """Preprocesses covariates and response and generates training, validation and testing sets.
    
      Features are processed as follows:
      Missing values are imputed using the mean. After imputation, all features are standardized. 

      Responses are processed as follow:
      Either standardized or not depending on user choice selected by standardize_response.

    Args:
        val_ratio: Percentage of samples to be used for validation, float scalar.
        test_ratio: Percentage of samples to be used for testing, float scalar.
        seed: for reproducibility of results, int scalar.
        standardize_response: whether to standardize target response or not, bool scalar.
        
    Returns:
        X_train: Training processed covariates, float numpy array of shape (N, p).
        y_train: Training (processed) responses, float numpy array of shape (N, ).
        X_val: Validation processed covariates, float numpy array of shape (Nval, p).
        y_val: Validation (processed) responses, float numpy array of shape (N, ).
        X_test: Test processed covariates, float numpy array of shape (Ntest, p).
        y_test: Test (processed) responses, float numpy array of shape (N, ).
        x_preprocessor: processor for covariates, sklearn transformer.
        y_preprocessor: processor for responses, sklearn transformer.
    """        
#     house_values_variables = []
#     for i in df_X.columns:
#         if 'House_Value' in i:
#             house_values_variables.append(i)        
#     df_X[house_values_variables] = 100*(df_X[house_values_variables]/(df_X[house_values_variables].max(axis=0)))
    
#     income_variables = []
#     for i in df_X.columns:
#         if 'INC' in i or 'Inc' in i:
#             income_variables.append(i)        
#     df_X[income_variables] = 100*(df_X[income_variables]/(df_X[income_variables].max(axis=0)))
        
    N, p = df_X.shape
    df_X_temp, df_X_test, df_y_temp, df_y_test = train_test_split(df_X, df_y, test_size=int(test_ratio*N), random_state=seed)
    df_X_train, df_X_val, df_y_train, df_y_val = train_test_split(df_X_temp, df_y_temp, test_size=int(val_ratio*N), random_state=seed)
    
    print("Number of training samples:", df_X_train.shape[0])
    print("Number of validation samples:", df_X_val.shape[0])
    print("Number of test samples:", df_X_test.shape[0])
    print("Number of covariates:", p)
        
    ''' Processing Covariates '''    
    continuous_features = df_X.columns
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    x_preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features)])

    X_train = x_preprocessor.fit_transform(df_X_train)
    X_val = x_preprocessor.transform(df_X_val)
    X_test = x_preprocessor.transform(df_X_test)
    
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)    
    X_train = np.round(X_train, decimals=6)
    X_val = np.round(X_val, decimals=6)
    X_test = np.round(X_test, decimals=6)
    
    ''' Processing Target Responses '''
    if standardize_response:
        y_preprocessor = StandardScaler()
    else:
        def identity_func(x):
            return np.array(x)
        y_preprocessor = FunctionTransformer(lambda x: np.array(x)) # acts as identity

    y_train = y_preprocessor.fit_transform(df_y_train)
    y_val = y_preprocessor.transform(df_y_val)
    y_test = y_preprocessor.transform(df_y_test)
                
    return X_train, y_train, X_val, y_val, X_test, y_test, (x_preprocessor, x_scaler), y_preprocessor


parser = argparse.ArgumentParser(description='GAMINet on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--path', dest='path',  type=str, default='/home/gridsan/shibal')
parser.add_argument('--version', dest='version',  type=int, default=1)

args = parser.parse_args()


load_directory=args.load_directory
# save_directory = os.path.join(os.path.abspath(os.getcwd()).split('src')[0], "results") 

df_X, df_y, _ = load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
seed = args.seed
np.random.seed(seed)
X, Y, Xval, Yval, Xtest, Ytest, _, y_scaler = process_data(
    df_X,
    df_y,
    val_ratio=0.1, 
    test_ratio=0.1,
    seed=seed,
    standardize_response=False)

save_directory = os.path.join(args.path, "elaan/baselines/GamiNet/examples", "results", "Census", "v{}/seed{}".format(args.version, args.seed))
os.makedirs(save_directory, exist_ok=True)

task_type = 'Regression'
meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(X.shape[1])}
meta_info.update({'Y':{'type':'target'}})         
get_metric = metric_wrapper(mse, y_scaler)

for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        def identity_func(x):
            return np.array(x)
        sy = FunctionTransformer(lambda x: np.array(x)) # acts as identity
#             sy = MinMaxScaler((0, 1))
        Y = sy.fit_transform(Y)
        Yval = sy.transform(Yval)
        Ytest = sy.transform(Ytest)
    
        meta_info[key]['scaler'] = sy
    else:
        sx = MinMaxScaler((0, 1))
        sx.fit([[0], [1]])
        X[:,[i]] = sx.transform(X[:,[i]])
        Xval[:,[i]] = sx.transform(Xval[:,[i]])
        Xtest[:,[i]] = sx.transform(Xtest[:,[i]])
        meta_info[key]['scaler'] = sx


# meta_info

## Train GAMI-Net 
start = time.time()

model = GAMINet(
    meta_info=meta_info,
    interact_num=500, 
    interact_arch=[40] * 5,
    subnet_arch=[40] * 5, 
    batch_size=200,
    task_type=task_type,
    activation_func=tf.nn.relu, 
    main_effect_epochs=500,
    interaction_epochs=500,
    tuning_epochs=500, 
    lr_bp=[0.0001, 0.0001, 0.0001],
    early_stop_thres=[50, 50, 50],
    heredity=True,
    loss_threshold=0.001,
    reg_clarity=1,
    verbose=True,
    val_ratio=0.1,
#     random_state=random_state
)
model.fit(X, Y, valid_x=Xval, valid_y=Yval)

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(save_directory, "Tuning.txt"), "a") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} hyperparameter settings.\n".format(int(hours), int(minutes), seconds, 1))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds, 1)) 

pred_train = model.predict(X)
pred_val = model.predict(Xval)
pred_test = model.predict(Xtest)
gaminet_stat = np.hstack(
    [
        np.round(get_metric(Y, pred_train), 5), 
        np.round(get_metric(Yval, pred_val), 5),
        np.round(get_metric(Ytest, pred_test), 5)
    ]
)
print(gaminet_stat)


mse_train, rmse_train = metrics(Y, model.predict(X))
mse_valid, rmse_valid = metrics(Yval, model.predict(Xval))
mse_test, rmse_test = metrics(Ytest, model.predict(Xtest))

pen = np.array(['GAMINet'])
M = pen.shape[0]
df = pd.DataFrame(data={'': pen,
                      'Training MSE': np.zeros(M), 
                      'Validation MSE': np.zeros(M),
                      'Test MSE': np.zeros(M),
                      'Test RMSE': np.zeros(M),
                      'Nonzeros': np.zeros(M)})
df = df.set_index('')


print("Train: RMSE:", rmse_train)
print("Val: RMSE:", rmse_valid)
print("Test: RMSE:", rmse_test)



interaction_effects = []
for indice in model.active_interaction_index:
    inter_net = model.interact_blocks.interacts[indice]
    feature_name1 = model.feature_list_[model.interaction_list[indice][0]]
    feature_name2 = model.feature_list_[model.interaction_list[indice][1]]
    print(feature_name1, feature_name2)
    interaction_effects.append((feature_name1, feature_name2))

interaction_effects = np.array([interaction_effects])

df.loc['GAMINet','Training MSE'] = mse_train
df.loc['GAMINet','Validation MSE'] = mse_valid
df.loc['GAMINet','Test MSE'] = mse_test
df.loc['GAMINet','Test RMSE'] = rmse_test
df.loc['GAMINet','Nonzeros']=len(np.unique(np.concatenate([model.active_main_effect_index, np.array([i.split("X")[1] for i in np.unique(interaction_effects)]).astype(np.int64)])))
df.loc['GAMINet','MainEffects']=len(model.active_main_effect_index)
df.loc['GAMINet','InteractionEffects']=len(model.active_interaction_index)

with open(os.path.join(save_directory, 'results.csv'), 'a') as f:
    df.loc[['GAMINet'],:].to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)


# ## Visualization

# # Training details

# data_dict_logs = model.summary_logs(save_dict=False)
# plot_trajectory(data_dict_logs, folder=save_directory, name="s1_traj_plot", log_scale=True, save_png=True, save_eps=False)
# plot_regularization(data_dict_logs, folder=save_directory, name="s1_regu_plot", log_scale=True, save_png=True, save_eps=False)

# # Global Visualization

# data_dict_global = model.global_explain(save_dict=False)
# global_visualize_density(data_dict_global, save_png=True, folder=save_directory, name='s1_global')

# # Feature Importance

# feature_importance_visualize(data_dict_global, save_png=True, folder=save_directory, name='s1_feature')
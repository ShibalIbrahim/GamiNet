import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def simu_loader(data_generator, datanum, testnum, noise_sigma):
    def wrapper(rand_seed=0):
        return data_generator1(datanum, testnum=testnum, noise_sigma=noise_sigma, rand_seed=rand_seed)
    return wrapper


def dataset_loader(data_loder, path, test_ratio):
    def wrapper(rand_seed=0):
        return data_generator1(path, test_ratio=test_ratio, rand_seed=rand_seed)
    return wrapper


def data_generator1(datanum, testnum=10000, noise_sigma=1, rand_seed=0):
    
    np.random.seed(rand_seed)
    x = np.zeros((datanum + testnum, 10))
    for i in range(10):
        x[:, i:i+1] = np.random.uniform(-1,1,[datanum + testnum,1])
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x[:, [i]] for i in range(10)]

    def cliff(x1, x2):
        # x1: -20,20
        # x2: -10,5
        x1 = x1 * 20
        x2 = x2 * 7.5 - 2.5
        term1 = -0.5 * x1 ** 2 / 100
        term2 = -0.5 * (x2 + 0.03 * x1 ** 2 - 3) ** 2
        y = np.exp(term1 + term2)
        return  y

    y = (2 * x1 ** 2
         + 0.2 * np.exp(-4 * x2)
         + 2.5 * np.sin(np.pi * x3 * x4)
         + 5 * cliff(x5, x6)).reshape([-1,1]) + noise_sigma*np.random.normal(0, 1, [datanum + testnum, 1])

    task_type = "Regression"
    meta_info = {"X1":{"type":"continuous"},
             "X2":{"type":"continuous"},
             "X3":{"type":"continuous"},
             "X4":{"type":"continuous"},
             "X5":{"type":"continuous"},
             "X6":{"type":"continuous"},
             "X7":{"type":"continuous"},
             "X8":{"type":"continuous"},
             "X9":{"type":"continuous"},
             "X10":{"type":"continuous"},
             "Y":{"type":"target"}}
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            x[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=testnum, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def data_generator2(datanum, testnum=10000, noise_sigma=1, rand_seed=0):
    np.random.seed(rand_seed)
    x = np.random.uniform(0, 1, [datanum + testnum, 6])
    x1, x2, x3, x6, x7, x9 = [x[:, [i]] for i in range(6)]
    x = np.random.uniform(0.6, 1, [datanum + testnum, 4])
    x4, x5, x8, x10 = [x[:, [i]] for i in range(4)]
    x = np.hstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    y = (np.pi**(x1 * x2) * np.sqrt(2 * x3) 
          - np.sin(x4)**(-1) 
          + np.log10(x3 + x5) 
          - x9 / x10 * np.sqrt(x7 / x8) 
          - x2 * x7) + noise_sigma * np.random.normal(0, 1, [datanum + testnum, 1])

    task_type = "Regression"
    meta_info = {"X1":{"type":"continuous"},
             "X2":{"type":"continuous"},
             "X3":{"type":"continuous"},
             "X4":{"type":"continuous"},
             "X5":{"type":"continuous"},
             "X6":{"type":"continuous"},
             "X7":{"type":"continuous"},
             "X8":{"type":"continuous"},
             "X9":{"type":"continuous"},
             "X10":{"type":"continuous"},
             "Y":{"type":"target"}}
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            x[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=testnum, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_wine(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "wine/winequality-white.csv", sep=";")
    meta_info = json.load(open(path + "wine/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values

    xx = np.zeros(x.shape)
    task_type = "Regression"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y.astype(np.float32), test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_concrete(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_excel(path + "concrete/Concrete_Data.xls")
    meta_info = json.load(open(path + "concrete/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    
    xx = np.zeros(x.shape)
    task_type = "Regression"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y.astype(np.float32), test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_skill_craft(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "skill_craft/SkillCraft1_Dataset.csv").replace("?", np.nan).dropna()
    meta_info = json.load(open(path + "skill_craft/data_types.json"))
    x, y = data.iloc[:,2:].values, data.iloc[:,[1]].values
    
    xx = np.zeros(x.shape)
    task_type = "Regression"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y.astype(np.float32), test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_parkinsons(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "parkinsons_tele/parkinsons_updrs.data", index_col=[0])
    meta_info = json.load(open(path + "parkinsons_tele/data_types.json"))
    x, y = pd.concat([data.iloc[:,:3], data.iloc[:,5:]], 1).values, data.loc[:,['total_UPDRS']].values

    xx = np.zeros(x.shape)
    task_type = "Regression"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y.astype(np.float32), test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


## Classification
def load_magic(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "magic/magic04.data", header=None, sep=",")
    meta_info = json.load(open(path + "magic/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].replace("g", 1).replace("h", 0).values
    
    xx = np.zeros(x.shape)
    task_type = "Classification"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_spambase(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "spambase/spambase.data", header=None, sep=",")
    meta_info = json.load(open(path + "spambase/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    
    xx = np.zeros(x.shape)
    task_type = "Classification"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_seismic_bumps(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "seismic_bumps/seismic-bumps.arff")
    meta_info = json.load(open(path + "seismic_bumps/data_types.json"))
    x, y = np.hstack([data.iloc[:,:13].values, data.iloc[:,16:-1].values]), data.iloc[:,[-1]].values

    xx = np.zeros(x.shape)
    task_type = "Classification"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_credit_default(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_excel(path + "credit_default/default of credit card clients.xls", header=1)
    meta_info = json.load(open(path + "credit_default/data_types.json"))
    payment_list = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
               'PAY_AMT1', 'PAY_AMT2','PAY_AMT3', 'PAY_AMT4','PAY_AMT5', 'PAY_AMT6']
    data.loc[:,payment_list] = (np.sign(data.loc[:,payment_list]).values * np.log10(np.abs(data.loc[:,payment_list]) + 1))
    x, y = data.iloc[:,1:-1].values, data.iloc[:,[-1]].values

    xx = np.zeros(x.shape)
    task_type = "Classification"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def load_fico_challange(path="./data/", test_ratio=0.2, rand_seed=0):
    data = pd.read_csv(path + "fico_challenge/heloc_dataset_v1.csv")
    meta_info = json.load(open(path + "fico_challenge/data_types.json"))
    data = data.replace(-9, np.nan).replace(-8, np.nan).replace(-7, np.nan)

    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(data.iloc[:,1:])  
    data.iloc[:,1:] = imp.transform(data.iloc[:,1:])
    x, y = data.iloc[:,1:].values, data.iloc[:,[0]].values

    xx = np.zeros(x.shape)
    task_type = "Classification"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((-1, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    return train_x, test_x, train_y, test_y, task_type, meta_info

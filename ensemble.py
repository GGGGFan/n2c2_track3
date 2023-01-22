import pandas as pd
import pickle
import re
from collections import Counter
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import mode
from sklearn.model_selection import KFold, cross_val_score

if __name__ == "__main__":
    # load files for training ensembler
    res_1_dev = np.load('pred_model_1_dev.npy')
    res_2_dev = np.load('pred_model_2_dev.npy')
    res_3_dev = np.load('pred_model_3_dev.npy')
    res_4_dev = np.load('pred_model_4_dev.npy')
    res_5_dev = np.load('pred_model_5_dev.npy')
    res_6_dev = np.load('pred_model_6_dev.npy')
    df_dev = pd.read_csv("/home/jifangao/N2C2_track3/data/N2C2-Track3-May3/dev.csv")
    df_test = pd.read_csv('/home/jifangao/N2C2_track3/data/n2c2_track3_test/n2c2_track3_test.csv')
    # load files for final predictions
    res_1_test = np.load('pred_model_1_test.npy')
    res_2_test = np.load('pred_model_2_test.npy')
    res_3_test = np.load('pred_model_3_test.npy')
    res_4_test = np.load('pred_model_4_test.npy')
    res_5_test = np.load('pred_model_5_test.npy')
    res_6_test = np.load('pred_model_6_test.npy')
    ary_fts_dev = np.load("/home/jifangao/N2C2_track3/added_fts_dev_1010.npy")
    ary_fts_te = np.load("/home/jifangao/N2C2_track3/added_fts_te_1010.npy")
    # true labels
    dic_label_index = {'Direct': 0, 'Indirect': 1, 'Neither': 2, 'Not Relevant': 3}
    y_true_dev = np.array([dic_label_index[row.Relation] for _, row in df_dev.iterrows()])
    y_true_test = np.array([dic_label_index[row.Relation] for _, row in df_test.iterrows()])
    # build ensembler
    params = {
        'objective': 'multiclass',
        "max_depth": 2,
        'num_leaves': 31,
        "learning_rate" : 0.03,
        'min_data_in_leaf': 32,
        'num_iterations': 100,
        'lambda_l1': 0.02,
        'lambda_l2': 0.02,
        'num_classes': 4
    }
    params = best_params

    X_meta_dev = np.concatenate([ary_fts_dev,res_1_dev,res_2_dev,res_3_dev,res_4_dev,res_5_dev,res_6_dev],
                                  axis=1)
    X_meta_test = np.concatenate([ary_fts_te,res_1_test,res_2_test,res_3_test,res_4_test,res_5_test,res_6_test],
                                  axis=1)
    lgb_train = lgb.Dataset(X_meta_dev, label=y_true_dev, feature_name=ft_names)
    lgb_model = lgb.train(params, lgb_train)
    y_pred = np.argmax(lgb_model.predict(X_meta_test), axis=1)
    print(f"Macro F1: {round(f1_score(y_true_test, y_pred, average='macro'), 3)}")

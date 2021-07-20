import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

import pandas as pd
import numpy as np
import pickle
import random
from modeltraining import model_training
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


## Read datasets
### Internal data
df_cov = pd.read_csv('./harbor_scaled.csv')
df_cov = df_cov.fillna('N/A')

## External: Miami dataset

df_miami = pd.read_csv('./Miami_consolidated.csv')
df_miami = df_miami.fillna('N/A')


mon = [3,6,9,12,15,18,21]
for m in mon:
    print(m)
    fpr_CV = dict()
    tpr_CV = dict()
    roc_auc_CV = dict()
    prediction = dict()
    gt = dict()
    precision = dict()
    recall = dict()
    roc_pr_CV = dict()
    strm = 'Outcome at '+str(m)+' months'
    df_cov9 = df_cov[df_cov[strm]!='N/A']
    df_miami = df_miami[df_miami[strm]!='N/A'] 
    
    print(len(df_cov9['Patient number'].unique()))
    #for fold in range(1,11):
    NN = [5, 10, 20, 25, 30, 50]
    flg = [0,1]
    for f in flg:

        for n in NN:
            fold = random.randint(1,10)
            print('fold'+str(fold))
            fpr,tpr,roc_auc,ind_preds3mon,ind_Y_test3mon, lr_precision, lr_recall, lr_auc = model_training(df_cov9,df_miami, m, fold, n, f, strm)
            fpr_CV[n] = fpr
            tpr_CV[n] = tpr
            roc_auc_CV[n] = roc_auc
            prediction[n] = ind_preds3mon[ind_Y_test3mon[:,2]==0,1]
            gt[n] = ind_Y_test3mon[ind_Y_test3mon[:,2]==0,1]
            precision[n] = lr_precision
            recall[n] = lr_recall
            roc_pr_CV[n] = lr_auc
            print(roc_auc)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_predictionv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_fprv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(fpr_CV, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/ Miami'+str(m)+'mon_tprv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(tpr_CV, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_roc_aucv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(roc_auc_CV, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_GTv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_precisionv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(precision, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/ Miami'+str(m)+'mon_recallv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(recall, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./CV_resultsv2/Miami'+str(m)+'mon_roc_prv7_loss_'+str(f)+'.pickle', 'wb') as handle:
                pickle.dump(roc_pr_CV, handle, protocol=pickle.HIGHEST_PROTOCOL)
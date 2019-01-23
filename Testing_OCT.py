import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.layers.convolutional import ZeroPadding2D
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import keras as k


def training_data(df_cov, outcomestring):
    n_patient  = len(df_cov['Patient number'].unique())
    patient_ids = df_cov['Patient number'].unique()
    patients_vec = []
    patients_label = []
    Seq_len =[]
    print(n_patient)
    for i in range(n_patient):
        patients_vec.append([])
        patients_label.append([])
        p = df_cov.loc[df_cov['Patient number'] == patient_ids[i]]
        p = p.sort_values(['stop'], ascending=[True])
        Seq_len.append(len(p))
        #store vector and labels for 
        for j in range(len(p)):
            temp = p.iloc[j]
            temp = temp.fillna(0)
            patients_label[i].append(p.iloc[j][outcomestring])
            temp = temp.drop("Patient number")
            temp = temp.drop("Event")
            temp = temp.drop("stop")
            temp = temp.drop("start")
            temp = temp.drop("Outcome at 6 months") 
            temp = temp.drop("Outcome at 3 months")
            temp = temp.drop("Outcome at 9 months") 
            temp = temp.drop("Outcome at 12 months")
            temp = temp.drop("Outcome at 15 months") 
            temp = temp.drop("Outcome at 18 months")
            temp = temp.drop("Outcome at 21 months")
            temp = temp.drop("Elapsed time since first imaging")
            patients_vec[i].append(temp.tolist())
    return temp, patients_vec, patients_label, Seq_len



def create_model(slen, num_features):   
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='sigmoid', stateful=False, input_shape=(slen, num_features), name='LSTM_1'))
    model.add(BatchNormalization())
    model.add(LSTM(20, return_sequences=True, activation='sigmoid', stateful=False, name='LSTM_2'))
    model.add(TimeDistributed(Dense(3, activation='softmax', name='Softmax'),name ='TimeDis_main_output')) 
    return model


## run the testing - read small sample dataset (fold 1 in the paper)
test = pd.read_csv('./weights/sample_data.csv')


## 3months prediction
test = test.fillna('N/A')
test1  = test[test['Outcome at 3 months']!='N/A']
test1 = test1.drop(['Unnamed: 0'], axis=1)
test1 = test1.drop(['Unnamed: 0.1'], axis=1)
temp, patients_vec_test, patients_label_test, Seq_len_test = training_data(test1, "Outcome at 3 months")
slen = 25
  
X_test = pad_sequences(patients_vec_test, slen, padding='pre', truncating='pre', value=0, dtype='float32')
Y_test = pad_sequences(patients_label_test, slen, padding='pre', truncating='pre', value=2.)
num_features = X_test.shape[2]

Y_categorical_test = k.utils.to_categorical(Y_test, 3)
Y_categorical_test = Y_categorical_test.reshape(Y_test.shape[0], Y_test.shape[1], 3)
y_test = Y_categorical_test

## Load trained weights
wei = './weights/3monweights.h5py'
bestmodel = create_model(slen, num_features)
bestmodel.load_weights(wei)
batch_size = 50
preds_prob3mon = bestmodel.predict_proba(X_test, batch_size=batch_size)
preds_prob3mon.shape
ind_preds3mon = preds_prob3mon.reshape(X_test.shape[0]*slen,3)
ind_Y_test3mon = y_test.reshape(X_test.shape[0]*slen,3)

## ROC with out padding 
fpr3, tpr3,thresholds = roc_curve(np.array(ind_Y_test3mon[ind_Y_test3mon[:,2]==0,1]), np.array(ind_preds3mon[ind_Y_test3mon[:,2]==0,1]))
roc_auc3 = auc(fpr3, tpr3)

    
## 21 months
test2 = test[test['Outcome at 21 months']!='N/A']
test2 = test2.drop(['Unnamed: 0'], axis=1)
test2 = test2.drop(['Unnamed: 0.1'], axis=1)
temp, patients_vec_test, patients_label_test, Seq_len_test = training_data(test2, "Outcome at 21 months")
slen = 22
  
X_test = pad_sequences(patients_vec_test, slen, padding='pre', truncating='pre', value=0, dtype='float32')
Y_test = pad_sequences(patients_label_test, slen, padding='pre', truncating='pre', value=2.)
num_features = X_test.shape[2]

Y_categorical_test = k.utils.to_categorical(Y_test, 3)
Y_categorical_test = Y_categorical_test.reshape(Y_test.shape[0], Y_test.shape[1], 3)
y_test = Y_categorical_test

wei = './weights/21monweights.h5py'
bestmodel = create_model(slen, num_features)
bestmodel.load_weights(wei)
batch_size = 50
preds_prob3mon = bestmodel.predict_proba(X_test, batch_size=batch_size)
preds_prob3mon.shape
ind_preds21mon = preds_prob3mon.reshape(X_test.shape[0]*slen,3)
ind_Y_test21mon = y_test.reshape(X_test.shape[0]*slen,3)

## ROC with out padding 
fpr21, tpr21,thresholds = roc_curve(np.array(ind_Y_test21mon[ind_Y_test21mon[:,2]==0,1]), np.array(ind_preds21mon[ind_Y_test21mon[:,2]==0,1]))
roc_auc21 = auc(fpr21, tpr21)

    
## Plot teh ROC curve
plt.figure()
lw = 2
plt.plot(fpr3, tpr3, color='darkorange',
         lw=lw, label='ROC curve for 3 months (area = %0.2f)' % roc_auc3)
plt.plot(fpr21, tpr21, color='darkred',
         lw=lw, label='ROC curve for 21 months (area = %0.2f)' % roc_auc21)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for short and long term prediction')
plt.legend(loc="lower right")
plt.show()




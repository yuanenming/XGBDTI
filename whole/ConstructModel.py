#

import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics



#load data
X = pd.read_table('totalX.txt',sep = '\t', header = None)
interMatrix = np.loadtxt('mat_p_d_Refined.txt')
sp = interMatrix.shape
Y = interMatrix.T.reshape((-1,1))
X = X.iloc[:,:]

######################################################
## This part is training the model. I have finished  #
## The model is saved into big.model                 #
######################################################

# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,random_state=1)

# # convert to xgboost data structure
# data_train = xgb.DMatrix(x_train, label = y_train)
# data_test = xgb.DMatrix(x_test, label = y_test)

# training
# params = {
#     'booster': 'gbtree',
#     'objective': 'binary:logistic',  
#     'gamma': 0.00,
#     'max_depth': 9,
#     'lambda': 1, 
#     'colsample_bylevel':0.7,
#     'subsample': 0.8, 
#     'colsample_bytree': 0.7, 
#     'min_child_weight': 1,
#     'silent': 0, 
#     'eta': 0.1, 
#     'seed': 1,
#     'eval_metric': "auc",
#     'learning_rate':0.15,
#     'scale_pos_weight' : 10,
#     'n_estimators': 500,
#     'base_score':0.5,
#     'random_state':12,
#     'scoring' : 'roc_auc'
#     }
# watchlist = [(data_test, 'eval'), (data_train, 'train')]

# def evalAUPR(preds, dtrain):       
#     '''
#     This is an aupr evaluation function for XGBoost model.
#     Since there is no such things in XGBoost.
#     So I had to writen by myself.
#     '''
#     labels = dtrain.get_label()
#     precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
#     return 'aupr', metrics.auc(recall, precision)

# bst = xgb.train(params10, data_train, 100, watchlist, feval = evalAUPR)
# bst.save_model('big.model')
# print 'the model has been saved'
# exit()

# run
# dtrain = xgb.DMatrix(X, label = Y)
# bst = xgb.train(params, dtrain, 100, feval = evalAUPR)
# bst.save_model('big.model')

# dtrain = xgb.DMatrix(X, label = Y)
# bst = xgb.Booster({'nthread':4}) #init model
# bst.load_model("big.model") # load data
# ypred = bst.predict(dtrain)
# ypred = ypred.reshape(sp)

# xhat, yhat = np.where(ypred > 0.5)
# print xhat.shape
# predDTIs = []
# for i in xrange(len(xhat)):
#     predDTIs.append((xhat[i],yhat[i]))
# x, y = np.where(interMatrix == 1)
# knownDTIs = []
# for i in xrange(len(x)):
#     knownDTIs.append((x[i],y[i]))
# newDTIs = np.array(list(set(predDTIs) - set(knownDTIs)))
# print newDTIs
# print newDTIs.shape





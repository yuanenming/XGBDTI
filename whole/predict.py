import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################
## Predict new DTIs.                                 #
######################################################


X = pd.read_table('totalX.txt',sep = '\t', header = None)
interMatrix = np.loadtxt('mat_p_d_Refined.txt')
sp = interMatrix.shape
Y = interMatrix.T.reshape((-1,1))
X = X.iloc[:,:]

dtrain = xgb.DMatrix(X, label = Y)
bst = xgb.Booster({'nthread':4}) #init model
bst.load_model("big.model") # load data
ypred = bst.predict(dtrain)
ypred = ypred.reshape(sp)

xhat, yhat = np.where(ypred > 0.5)
print xhat.shape
predDTIs = []
for i in xrange(len(xhat)):
    predDTIs.append((xhat[i],yhat[i]))
x, y = np.where(interMatrix == 1)
knownDTIs = []
for i in xrange(len(x)):
    knownDTIs.append((x[i],y[i]))
newDTIs = np.array(list(set(predDTIs) - set(knownDTIs)))
print newDTIs
#!/usr/bin/python2
# -*- encoding:utf-8 -*-
#
# Author: Enming Yuan

########### IMPORTS ##########################################################
import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
import scipy.spatial.distance as dist
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import metrics
import tuning_xgboost

#############  FUNCTION DEFINITIONS ##########################################
def computeSimilarity(netsList):
    '''
    Compute the Jaccard similarity matrix from the input heterogeneous networks.
    And write them to files.
    '''
    print 'Computing Similarity Matrix ...'
    for i in netsList:
        print i
        inputID = '../Data/InteractionData/' + i + '.txt'
        mat = np.loadtxt(inputID)
        similarity = dist.pdist(mat,'jaccard')
        similarity = 1 - dist.squareform(similarity)
        similarity[scipy.isnan(similarity)] = 0
        outputID = '../network/Sim_' + i + '.txt'
        np.savetxt(outputID, similarity, '%.6f', delimiter = '\t')

    mat = np.loadtxt('../Data/SimilarityData/Similarity_Matrix_Drugs.txt')
    np.savetxt('../network/Sim_mat_Drugs.txt', mat, '%.6f', delimiter = '\t')
    mat = np.loadtxt('../Data/SimilarityData/Similarity_Matrix_Proteins.txt')
    np.savetxt('../network/Sim_mat_Proteins.txt', mat, '%.6f', delimiter = '\t')
    return 

def RWR(network, maxIterator, restartProbability):
    '''
    Implamentation of Random Walk with Restart.
    Test scale free
    '''
    print 'Run Random Walk with Restart ...'
    nDimension = network.shape[0]

    # Add self-edge to isolated nodes
    network = network + np.diag(sum(network) == 0)
    # Normalize the adjacency matrix
    normNetwork = preprocessing.normalize(network, 'l1', axis = 0)
    restart = np.eye(nDimension)
    S = np.eye(nDimension)

    # Random work process
    for i in xrange(maxIterator):
        S_new = (1 - restartProbability) * np.dot(normNetwork, S) + np.dot(restartProbability, restart)
        delta = np.linalg.norm(S - S_new, ord = 'fro')
        S = S_new
        if delta < 1e-6:
            break
    return S

def PCAmodel(networksList, dimension, restartProbability, maxIterator):
    '''
    Run Random Walk with Restart first. Then run PCA on diffusion states.
    '''
    for i in xrange(len(networksList)):
        fileID = '../network/' + networksList[i] + '.txt'
        network = np.loadtxt(fileID)
        tQ = RWR(network, maxIterator, restartProbability)
        if not i:
            Q = tQ
        else:
            Q = np.concatenate([Q, tQ], axis = 1)

    # print 'Run PCA ...'
    # pca = PCA(n_components=dimension, whiten=True, random_state=0)
    # X = pca.fit_transform(Q)
    # return X
    print 'Run PCA ...'
    nNode = Q.shape[0]
    alpha = 1.0 / nNode
    Q = np.log(Q + alpha) - np.log(alpha)

    Q = np.dot(Q, np.transpose(Q))
    U, S, VT = scipy.sparse.linalg.svds(Q, dimension)
    X = np.dot(U, np.sqrt(np.sqrt(np.diag(S))))
    return X

def constructFeatures(drugNetworkList, proteinNetworkList, restartProbability = 0.5,\
    maxIterator = 20,dimDrug = 100, dimProtein = 400):
    '''
    Extract features from two lists of network files.
    And write them to the ../feature directory.
    '''
    drugFeatures = PCAmodel(drugNetworkList, dimDrug, restartProbability, maxIterator)
    proteinFeatures = PCAmodel(proteinNetworkList, dimProtein, restartProbability, maxIterator)
    print 'Construct Features ...'
    np.savetxt('../feature/drug%d.txt'%dimDrug, drugFeatures, '%.6f', delimiter = '\t')
    np.savetxt('../feature/protein%d.txt'%dimProtein, proteinFeatures, '%.6f', delimiter = '\t')
    return drugFeatures, proteinFeatures

def constructDataSet(drugFeatures, proteinFeatures, interMatrix, k = 1):
    '''
    Construct Dataset: k denotes the negtive positive ratio.
    And write the data set to the ../DataSet directory
    '''
    print 'Construct Data Set ...'
    drugFeatures = np.asmatrix(drugFeatures)
    proteinFeatures = np.asmatrix(proteinFeatures)
    interMatrix = np.asmatrix(interMatrix)

    pair_of_interaction = interMatrix.nonzero()
    num_of_interaction = len(pair_of_interaction[1])
    noninter = interMatrix - 1
    pair_of_noninteraction = noninter.nonzero()
    num_of_noninter = k * num_of_interaction
    
    # Shuffle
    np.random.RandomState(10).shuffle(pair_of_noninteraction[0])
    np.random.RandomState(10).shuffle(pair_of_noninteraction[1])
    pair_of_noninteraction = (pair_of_noninteraction[0][:num_of_noninter], \
        pair_of_noninteraction[1][:num_of_noninter])
    
    Y = np.concatenate([np.ones(num_of_interaction),np.zeros(num_of_noninter)])
    np.savetxt("../DataSet/Y%d.txt"%k, Y.T, fmt = '%d')

    for i in xrange(num_of_interaction + num_of_noninter):
        if not i:
            X = np.concatenate((proteinFeatures[pair_of_interaction[0][i]],drugFeatures[pair_of_interaction[1][i]]), axis = 1)
        elif i < num_of_interaction:
            x = np.concatenate((proteinFeatures[pair_of_interaction[0][i]],drugFeatures[pair_of_interaction[1][i]]), axis = 1)
            X = np.concatenate((X,x))
        else:
            i -= num_of_interaction
            x = np.concatenate((proteinFeatures[pair_of_noninteraction[0][i]],drugFeatures[pair_of_noninteraction[1][i]]), axis = 1)
            X = np.concatenate((X,x))
    np.savetxt("../DataSet/X%d.txt"%k, X, fmt = '%.5f')
    return X, Y.T

def XGBoostGridSearch(X, Y, plotting = False):
    '''
    Greedy Grid Search for hyperparameters
    This is a very time-consuming process.
    So I strongly recommand that you use the hyperparameters that I have already tuned.
    '''
    grid1 = {'n_estimators' : [300,400,500,600,700], 'learning_rate' : [0.05,0.1,0.15,0.20]}
    grid2 = {'max_depth' : [3,5,7,9], 'min_child_weight' : [3,5,7,9], 'gamma':[0.01,0.05,0.1]}
    grid3 = {'colsample_bylevel' : [0.5,0.6,0.7, 0.8, 0.9], 'subsample' : [0.5, 0.6, 0.7,0.8]}
    grid4 = {'scale_pos_weight' : [1,2,5]}

    hyperlist_to_try = [grid1, grid2, grid3, grid4]
    gridsearch_params = {
                    'cv'        : 3,
                    #'scoring'   : 'roc_auc',
                    }
    bst = xgb.XGBClassifier()
    print 'Run Simple Parameter Tuning ...'
    tuned_estimator   = tuning_xgboost.grid_search_tuning(X, Y,hyperlist_to_try,bst,gridsearch_params,plotting)
    tuned_parameters  = tuned_estimator.get_params()

    for parameter in  tuned_parameters:
        print parameter, '\t\t',tuned_parameters[parameter] 

def evalAUPR(preds, dtrain):       
    '''
    This is an aupr evaluation function for XGBoost model.
    Since there is no such things in XGBoost.
    So I had to writen by myself.
    '''
    labels = dtrain.get_label()
    precision, recall, thresholds = metrics.precision_recall_curve(labels, preds)
    return 'aupr', metrics.auc(recall, precision)

def evalRecall(preds, dtrain):
    labels = dtrain.get_label()
    recall = sum(labels[preds > 0.5])*1.0/sum(labels)
    return 'recall', recall

def evalPrecision(preds, dtrain):
    labels = dtrain.get_label()
    precision = sum(labels[preds > 0.5])*1.0/sum(preds > 0.5)
    return 'precision', precision

def evalF1(preds, dtrain):
    labels = dtrain.get_label()
    recall = sum(labels[preds > 0.5])*1.0/sum(labels)
    precision = sum(labels[preds > 0.5])*1.0/sum(preds > 0.5)
    return 'F1', 2*precision*recall/(precision+recall)

def evalMCC(preds, dtrain):
    labels = dtrain.get_label()
    TP = sum(labels[preds > 0.5])
    FN = sum(labels[preds < 0.5]==0)
    TN = sum(labels[preds>0.5]==0)
    FP = sum(labels[preds<0.5])
    return 'MCC', (TP*FN-TN*FP)*1.0/np.sqrt(sum(labels)*sum(labels==0)*sum(preds>0.5)*sum(preds<0.5))


def evalAccuracy(preds, dtrain):
    labels = dtrain.get_label()
    return 'accuracy', sum(labels == (preds > 0.5))*1.0/len(preds)



# This is the pre-tuned hyperparameters for XGBoost model when sample negtive positive ratio is 1.
params1 = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  
    'gamma': 0.0,   
    'max_depth': 9,    
    'lambda': 1,       
    'colsample_bylevel':0.7,
    'subsample': 0.8,
    'colsample_bytree': 0.7, 
    'min_child_weight': 4,
    'silent': 1, 
    'eta': 0.1,
    'seed': 12,
    'eval_metric': "auc",
    'learning_rate':0.18,
    'scale_pos_weight' : 1,
    'n_estimators': 600,
    'base_score':0.5,
    'random_state':1,
    'scoring' : 'roc_auc'
    }

# This is the pre-tuned hyperparameters for XGBoost model when sample negtive positive ratio is 10.
params10 = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  
    'gamma': 0.00,
    'max_depth': 12,
    'lambda': 1, 
    'colsample_bylevel':0.7,
    'subsample': 0.8, 
    'colsample_bytree': 0.7, 
    'min_child_weight': 4,
    'silent': 1, 
    'eta': 0.1, 
    'seed': 1,
    'eval_metric': "auc",
    'learning_rate':0.18,
    'scale_pos_weight' : 10,
    'n_estimators': 800,
    'base_score':0.5,
    'random_state':12,
    'scoring' : 'f1'
    }

def XGB_cv(X, Y, params, filename = 'XGB'):
    '''
    XGBoost classifier
    '''
    data_train = xgb.DMatrix(X, label=Y)
    data_train = xgb.DMatrix(X, label = Y)
    bst = xgb.cv(params, data_train, num_boost_round=100, nfold = 10, feval = evalAUPR)
    #bst.save_model('%s.model'%filename)
    print bst.iloc[-1:,:]
    return bst
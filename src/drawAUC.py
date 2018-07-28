#!/usr/bin/python2
# -*- encoding:utf-8 -*-
#
# Author: Enming Yuan

# cross validation on pre-trained data set

########### IMPORTS ##########################################################

from utils import *
from optparse import OptionParser
import matplotlib.pyplot as plt




if __name__ == '__main__':

    parser = OptionParser(description = 'Trains/evaluates XGBDTI models.')
    parser.add_option('-r', '--r', default = 'one', help = 'negtive positive ratio')
    (opts, args) = parser.parse_args()
    if opts.r == 'one':
        X = pd.read_table('../DataSet/X1.txt',sep = ' ', header = None)
        Y = pd.read_table('../DataSet/Y1.txt',sep = ' ', header = None)
        X, Y = X.iloc[:,:], Y.iloc[:,0]
        Y = pd.Categorical(Y).codes
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)#,random_state=0)
        data_train = xgb.DMatrix(x_train, label=y_train)
        data_test  = xgb.DMatrix(x_test, label= y_test)
        watch_list = [(data_test, 'eval'), (data_train, "train")]
        bst = xgb.train(params1, data_train, num_boost_round=100)
        y_test_proba = bst.predict(data_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba)
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_test_proba)
        aupr = metrics.auc(recall, precision)
        plt.figure(facecolor='w')
        plt.plot(fpr, tpr, 'r-', lw=2, alpha=0.8, color='blue', label='XGBDTI:AUROC=0.9264')
        plt.plot(recall, precision, 'r-', lw=2, alpha=0.8, color='green', label='XGBDTI:AUPR=0.9350')
        plt.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate / Recall', fontsize=14)
        plt.ylabel('True Positive Rate / Precision', fontsize=14)
        plt.grid(b=True)
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
        #plt.title('ROC-AUC of XGBDTI Algorithm', fontsize=17)
        plt.show()



        #XGB_cv(X, Y, params1)
    elif opts.r == 'ten':
        X = pd.read_table('../DataSet/X10.txt',sep = ' ', header = None)
        Y = pd.read_table('../DataSet/Y10.txt',sep = ' ', header = None)
        X, Y = X.iloc[:,:], Y.iloc[:,0]
        Y = pd.Categorical(Y).codes
        XGB_cv(X, Y, params10)

    else:
        print "Wrong negtive positive ratio!"
        exit(1)


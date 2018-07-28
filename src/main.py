#!/usr/bin/python2
# -*- encoding:utf-8 -*-
#
# Author: Enming Yuan

# cross validation on pre-trained data set

########### IMPORTS ##########################################################

from utils import *
from optparse import OptionParser



if __name__ == '__main__':

    parser = OptionParser(description = 'Trains/evaluates XGBDTI models.')
    parser.add_option('-r', '--r', default = 'one', help = 'negtive positive ratio')
    (opts, args) = parser.parse_args()
    print "running XGBDTI ..."
    if opts.r == 'one':
        X = pd.read_table('../DataSet/X1.txt',sep = ' ', header = None)
        Y = pd.read_table('../DataSet/Y1.txt',sep = ' ', header = None)
        X, Y = X.iloc[:,:], Y.iloc[:,0]
        Y = pd.Categorical(Y).codes
        XGB_cv(X, Y, params1)
    elif opts.r == 'ten':
        X = pd.read_table('../DataSet/X10.txt',sep = ' ', header = None)
        Y = pd.read_table('../DataSet/Y10.txt',sep = ' ', header = None)
        X, Y = X.iloc[:,:], Y.iloc[:,0]
        Y = pd.Categorical(Y).codes
        XGB_cv(X, Y, params10)

    else:
        print "Wrong negtive positive ratio!"
        exit(1)


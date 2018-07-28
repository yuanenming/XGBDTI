#!/usr/bin/python2
# -*- encoding:utf-8 -*-
#
# Author: Enming Yuan
# Total Pipeline for XGBDTI

########### IMPORTS ##########################################################

from utils import *
from optparse import OptionParser


if __name__ == '__main__':

    parser = OptionParser(description = 'Trains/evaluates NRNNMF models.')
    parser.add_option('-d', '--d', default = "Sim_mat_drug_drug,Sim_mat_drug_disease,Sim_mat_drug_se,Sim_mat_Drugs",\
         help = 'list of drug similarity networks in the ../network directory')
    parser.add_option('-p', '--p', default = "Sim_mat_protein_protein,Sim_mat_protein_disease,Sim_mat_Proteins",\
        help = 'list of protein similarity networks in the ../network directory')
    parser.add_option('-r', '--ratio', default = '1', help = 'negtive positive ratio: 1 or 10')
    parser.add_option('-m','--dm', default = '50', help = 'the dimension of drug feature vectors')
    parser.add_option('-n','--pm', default = '200', help = 'the dimension of protein feature vectors')
    (opts, args) = parser.parse_args()
    k = int(opts.ratio)


    ############ Compute Similarity ##########################################################
    Nets = ['mat_drug_drug', 'mat_drug_disease', 'mat_drug_se','mat_protein_protein', 'mat_protein_disease']
    computeSimilarity(Nets)

    ########### Feature Construction ##########################################################
    drugNets = list(opts.d.strip().split(','))
    proteinNets = opts.p.strip().split(',')
    dm = int(opts.dm)
    pm = int(opts.pm)
    drugFeatures, proteinFeatures = constructFeatures(drugNets, proteinNets, restartProbability = 0.5,\
        maxIterator = 20,dimDrug = dm, dimProtein = pm)

    ########### DataSet Construction ##########################################################
    k = int(opts.ratio)
    drugFeatures = np.loadtxt('../feature/drug%d.txt'%dm)
    proteinFeatures = np.loadtxt('../feature/protein%d.txt'%pm)
    interMatrix = np.loadtxt('../Data/InteractionData/mat_protein_drug.txt')
    X, Y = constructDataSet(drugFeatures, proteinFeatures, interMatrix, k)

    ########### Greedy Grid Search   ##########################################################
    # It is a veeeery time-consuming process. So use the hyperparameters I have already tuned
    
    # XGBoostGridSearch(X, Y)

    ########### Training and Evaluation ##########################################################
    X = pd.read_table('../DataSet/X%d.txt'%k,sep = ' ', header = None)
    Y = pd.read_table('../DataSet/Y%d.txt'%k,sep = ' ', header = None)
    X, Y = X.iloc[:,:], Y.iloc[:,0]
    Y = pd.Categorical(Y).codes
    if k is 1:
        XGB_cv(X, Y, params1)
    elif k is 10:
        XGB_cv(X, Y, params10)
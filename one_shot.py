
##Taken from https://github.com/GRAAL-Research/
##Modified by Vinod K Kurmi(vinodkk@iitk.ac.in)

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import scipy.io
import time
import copy
import os
import operator
import math
import json
import numpy as np
from sklearn.datasets import load_svmlight_files
from sklearn import svm
plt.ion()   # interactive mode
import h5py
adapt='yes'

#Load the source data feature in .mat file
if adapt=='yes':
    f_source = scipy.io.loadmat('Feature_Extrattor/features/Adapted_SourceDATA.mat')
    f_target = scipy.io.loadmat('Feature_Extrattor/features/Adapted_TargetDATA.mat')
elif adapt=='GRL':
    f_source = scipy.io.loadmat('Feature_Extrattor/features/GRL_SourceDATA.mat')
    f_target = scipy.io.loadmat('Feature_Extrattor/features/GRL_TargetDATA.mat')
else :
    f_source = scipy.io.loadmat('Feature_Extrattor/features/SourceDATA.mat')
    f_target = scipy.io.loadmat('Feature_Extrattor/features/TargetDATA.mat')
print 'data loading done'
dataSource = f_source.get('x')
dataSource = np.array(dataSource) # For converting to numpy array

#Load the Target data feature in .mat file
dataTarget = f_target.get('x')
dataTarget = np.array(dataTarget) # For converting to numpy array



def compute_proxy_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    # print source_X
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)

poe=compute_proxy_distance(dataSource,dataTarget)
print poe

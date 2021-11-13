import os,re,sys,math
import pandas as pd
import time
import datetime
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy import stats
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import import_data as data


train_path = 'data/dsjtzs_txfz_training.txt'
test_path = 'data/dsjtzs_txfz_test1.txt'

if __name__ == '__main__':
    train = data.read_data(train_path)
    train, label = train.drop(['id', 'label'], axis=1).astype(float), train['label'].values.astype(int)
    gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=50, min_samples_split=50, max_depth=9,
                                     min_samples_leaf=4, max_features=15, subsample=0.7)

    # train
    print('Start training...')
    gbm.fit(train, label)
    label_pred = gbm.predict(train)
    label_predprob = gbm.predict_proba(train)[:, 1]
    print("Accuracy: %.4f" % metrics.accuracy_score(label, label_pred))
    print("AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob))

    #test = data.read_data(test_path)
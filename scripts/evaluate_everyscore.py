#!/usr/bin/python
#
#  Copyright 2017 Hitomi Yanaka
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import glob
import numpy as np

from lxml import etree
from sklearn.externals import joblib
#import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
#import pylab as pl
#from matplotlib.backends.backend_pdf import *
#from sklearn.pipeline import make_pipeline
#from sklearn import preprocessing
#from sklearn.grid_search import GridSearchCV
import argparse

def rmse(x, y):
    ## x:targets y:predictions
    return np.sqrt(((y - x) ** 2).mean())

def load_pickle(results):
    with open('./'+results+'/all/features_np.pickle', 'rb') as in_f:
        train_sources = np.load(in_f)
        train_targets = np.load(in_f)
        trial_sources = np.load(in_f)
        trial_targets = np.load(in_f)
        train_id = np.load(in_f)
        trial_id = np.load(in_f)
    return train_sources, train_targets, trial_sources, trial_targets, train_id, trial_id
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    clf = joblib.load('./'+args.results+'/randomforestregressor.pkl')

    train_sources, train_targets, trial_sources, trial_targets, train_id, trial_id = load_pickle(args.results)
    trial_targets2 = [] #similarity_score
    for line in trial_id:
        #print(line)
        f = open('./plain2/sick_test_'+line+'.answer', 'r')
        score = f.readlines()[0].strip()
        trial_targets2.append(float(score))
    #extract only 1-5 results
    trial_targets2 = np.array(trial_targets2)
    trial_sources = np.c_[trial_sources, trial_targets2]

    f = open('./'+args.results+'/evaluate_eachscore.txt', 'w')
    for i in range(1, 5):
        lows, cols = np.where((trial_sources[:, 72:73]>i)&(trial_sources[:, 72:73]<=i+1))
        trial_sources_eval = trial_sources[lows,0:72]
        outputs = clf.predict(trial_sources_eval)
        trial_targets_eval = trial_targets2[lows]
        x = np.loadtxt(outputs, dtype=np.float32)
        y = np.loadtxt(trial_targets_eval, dtype=np.float32)
        r1, p1 = pearsonr(x, y)
        r2, p2 = spearmanr(x, y)
        r3 = rmse(x, y)
        f.write("{0}<i<={1} pearson: {2} spearman: {3} msr: {4}\n".format(i, i+1, r1, r2, r3))
    f.close()


if __name__ == '__main__':
    main()

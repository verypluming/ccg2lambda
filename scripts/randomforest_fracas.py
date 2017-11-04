#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import numpy as np
import scipy as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, f1_score
from deep_forest import MGCForest
import uuid
import sys
import matplotlib.pyplot as plt
import random
import difflib
from feature_extraction_fracas import *

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
import re
import argparse

from sklearn.cross_validation import cross_val_score

def crossvalidation(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    return scores.mean(), scores.std()

def regression(X_train=None, y_train=None, X_test=None, y_test=None, results="results"):
    parameters = {
        'n_estimators'      : [10, 50, 100, 200, 300],
        'random_state'      : [0],
        'n_jobs'            : [200],
        'max_features'      : ['auto', 'log2', 'sqrt', None],
        'criterion'         : ['mse'],
        'max_depth'         : [3, 5, 10, 20, 30, 40, 50, 100]
    }

    clf = make_pipeline(
        preprocessing.StandardScaler(),
        #preprocessing.MinMaxScaler(),
        GridSearchCV(RandomForestRegressor(), parameters))
    clf.fit(X_train, y_train)

    #Serialize
    joblib.dump(clf, './'+results+'/rte_fracas.pkl')
    #clf = joblib.load('results_20170614/randomforestclassifier_rte_wnw2v_sick.pkl')

    return clf

def deep_forest(X_train, y_train, X_test, y_test):
    mgc_forest = MGCForest(
        estimators_config={
            'mgs': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 200,
                    'min_samples_split': 2,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 200,
                    'min_samples_split': 2,
                    'n_jobs': -1,
                }
            }],
            'cascade': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }]
        },
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
    )
    mgc_forest.fit(X_train, y_train.astype(np.uint8))

    return mgc_forest

def get_features(line):
    if line[14] == "0":
        line[14] = "1"
    if line[22] == "0":
        line[22] = "1"
    unknown1, yes1, no1 = 0, 0, 0
    unknown2, yes2, no2 = 0, 0, 0
    if line[4] == "0":
        unknown1 = 1.0
        yes1 = 0.0
        no1 = 0.0
    elif line[4] == "0.5":
        unknown1 = 0.0
        yes1 = 0.0
        no1 = 1.0
    elif line[4] == "1":
        unknown1 = 0.0
        yes1 = 1.0
        no1 = 0.0
    if line[8] == "0":
        unknown2 = 1.0
        yes2 = 0.0
        no2 = 0.0
    elif line[8] == "0.5":
        unknown2 = 0.0
        yes2 = 0.0
        no2 = 1.0
    elif line[8] == "1":
        unknown2 = 0.0
        yes2 = 1.0
        no2 = 0.0
    sentence1_list = line[2].split()
    sentence2_list = line[3].split()
    features = [
        unknown1,
        yes1,
        no1,
        unknown2,
        yes2,
        no2,   
        float(line[5]),                     #6 axiom similarity A->B #6
        0.5**float(line[6]),                #7 axiom number A->B
        float(line[7]),                     #8 final subgoal A->B #8
        float(line[9]),                     #9 axiom similarity B->A #9
        0.5**float(line[10]),               #10 axiom number B->A
        float(line[11]),                    #11 final subgoal B->A #11
        float(line[12]),                    #12 original subgoal A->B
        float(line[13]),                    #13 original subgoal B->A
        float(line[14]),                    #14 step A->B
        float(line[15])/float(line[14]),    #15-21 inference rule A->B 
        float(line[16])/float(line[14]),
        float(line[17])/float(line[14]),
        float(line[18])/float(line[14]),
        float(line[19])/float(line[14]),
        float(line[20])/float(line[14]),
        float(line[21])/float(line[14]),
        float(line[22]),                    #22 step B->A
        float(line[23])/float(line[22]),    #23-29 inference rule B->A 
        float(line[24])/float(line[22]),
        float(line[25])/float(line[22]),
        float(line[26])/float(line[22]),
        float(line[27])/float(line[22]),
        float(line[28])/float(line[22]),
        float(line[29])/float(line[22]),
        float(line[30]),                    #30-35 subgoal case A->B, B->A #30-35
        float(line[31]),
        float(line[32]),
        float(line[33]),
        float(line[34]),
        float(line[35]),
        float(get_overlap(sentence1_list, sentence2_list)),                 #36 word overlap
        float(sentence_lengths(sentence1_list, sentence2_list)),            #37 sentence length
        float(difflib.SequenceMatcher(None, line[2], line[3]).ratio()),                        #38 string similarity
        float(word_overlap2(sentence1_list, sentence2_list)),               #39 Proportion of word overlap
        float(sentence_lengths_difference(sentence1_list, sentence2_list)), #40 Proportion of difference in sentence length
        float(synset_overlap(sentence1_list, sentence2_list)),              #41 Proportion of synset lemma overlap
        float(synset_distance(sentence1_list, sentence2_list)),             #42 Synset distance
        float(type_overlap(line[0])),                                       #43 type overlap
        float(pos_overlap(line[0])),                                        #44 pos-tag overlap
        float(noun_overlap(line[0])),                                       #45 Proportion of noun overlap
        float(verb_overlap(line[0])),                                       #46 Proportion of verb overlap
        float(pred_overlap(line[0])),                                       #47 Proportion of predicate overlap
        float(passive_overlap(line[0])),                                    #52 passive overlap 2017/02/14
        float(negation_overlap(line[0])),                                   #53 negation overlap 2017/02/14
    ]   
    return features

def rte2int(str):
    if str == "unknown":
        return 0.0
    elif str == "undef":
        return 0.0
    elif str == "no":
        return 1.0
    elif str == "yes":
        return 2.0

def retrieve_features(recalc=None, features=None, results=None):
    if recalc:
        # Extract training features and targets
        print ('Feature extraction')
        sources = np.array([get_features(line) for line in features])
        targets = np.array([float(rte2int(line[1])) for line in features])
        ids = np.array([line[0] for line in features])
        size = int(len(ids)/2)
        train_sources = sources[0:size]
        train_targets = targets[0:size]
        trial_sources = sources[size:]
        trial_targets = targets[size:]
        train_id = ids[0:size]
        trial_id = ids[size:]

        # Store to pickle for future reference
        with open('./'+results+'/all/features_np.pickle', 'wb') as out_f:
            np.save(out_f, train_sources)
            np.save(out_f, train_targets)
            np.save(out_f, trial_sources)
            np.save(out_f, trial_targets)
            np.save(out_f, train_id)
            np.save(out_f, trial_id)
    else:
        with open('./'+results+'/all/features_np.pickle', 'rb') as in_f:
            train_sources = np.load(in_f)
            train_targets = np.load(in_f)
            trial_sources = np.load(in_f)
            trial_targets = np.load(in_f)
            train_id = np.load(in_f)
            trial_id = np.load(in_f)
    return train_sources, train_targets, trial_sources, trial_targets, train_id, trial_id


def write_for_evaluation(outputs, trial_targets, ids, results):
    with open('./'+results+'/all_result_rte.txt', 'w') as out_f:
        out_f.write('pair_ID\tpred\tcorr\n')
        for i, line in enumerate(outputs):
            out_f.write('{0}\t{1}\t{2}\n'.format(ids[i], trial_targets[i], line))


def output_errors(outputs, trial_targets, ids, results):
    with open('./'+results+'/error_result_rte.txt', 'w') as out_f:
        out_f.write('pair_ID\tpred\tcorr\n')
        errs = []
        for i, line in enumerate(outputs):
            corr = trial_targets[i]
            if line != corr:
                errs.append([ids[i], line, corr])

        for err in errs:
            out_f.write('{0}\t{1}\t{2}\n'.format(*err))



def load_sick_data_from(filename, results):
    line = []
    line.append(filename)
    f = open('./fracas_plain/'+filename+'.answer', 'r')
    line.append(f.readlines()[0].strip())
    f.close()

    g = open('./fracas_plain/'+filename+'.txt', 'r')
    texts = g.readlines()
    line.append(texts[0].strip())
    line.append(texts[1].strip())
    g.close()
    if os.path.exists('./'+results+'/'+filename+'.answer'):
        h = open('./'+results+'/'+filename+'.answer', 'r')
        result = h.readlines()
        if result and not re.search("coq_error", result[0]) and not "unknown\n" in result:
            results = result[0].split(",")
            for r in results:
                r = r.strip("[] \n")
                line.append(r)
        else:
            return None
        h.close()
    else:
        return None

    i = open('./fracas_plain/'+filename+'.tok', 'r')
    texts = i.readlines()
    line.append(texts[0].strip())
    line.append(texts[1].strip())
    i.close()

    return line

def load_sick_data(results):
    sick_test = []
    filelist = glob.glob('./'+results+'/fracas*.answer')
    for filename in filelist:
        filename = re.search("./"+results+"/(.*?).answer", filename).group(1)
        if re.search("candc", filename):
            continue
        elif re.search("easyccg", filename):
            continue
        elif re.search("depccg", filename):
            continue
        if load_sick_data_from(filename, results) is not None:
            sick_test.append(load_sick_data_from(filename, results))
    return sick_test

## spearman correlation
def spearman(x, y):
    N = len(x)
    return 1 - (6 * sum(x - y) ** 2) / float(N**3 - N)

## root mean squared arror
def rmse(x, y):
    ## x:targets y:predictions
    return np.sqrt(((y - x) ** 2).mean())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    # Load sick data
    features = load_sick_data(args.results)
    random.seed(23)
    random.shuffle(features)
    
    # Get training and trial features
    train_sources, train_targets, trial_sources, trial_targets, train_id, trial_id = retrieve_features(1, features, args.results)
    #train_sources, train_targets, trial_sources, trial_targets = retrieve_features()
        
    # Train the regressor
    clf = regression(train_sources, train_targets, trial_sources, trial_targets, args.results)

    # Apply regressor to trial data
    outputs = clf.predict(trial_sources)

    # Evaluate regressor
    write_for_evaluation(outputs, trial_targets, trial_id, args.results) #Outputs and sick_ids

    # Check errors
    output_errors(outputs, trial_targets, trial_id, args.results) #Outputs and sick_ids

    #trial_targets = list(map(str, list(trial_targets)))
    #outputs = list(map(str, list(outputs)))
    f = open("fracas_results/rte_report.txt", "w")
    f.write(classification_report(trial_targets, outputs, digits=5))
    f.write("\n"+str(accuracy_score(trial_targets, outputs)))
    f.close()

    

if __name__ == '__main__':
    main()

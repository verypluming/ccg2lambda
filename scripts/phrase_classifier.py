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
import json
import glob
import re
import os
import numpy as np
import scipy as sp
import argparse
import random
import difflib

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, linear_model, svm
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

def crossvalidation(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    return scores.mean(), scores.std()

def base_model(activation="relu", optimizer="adam", out_dim=20):
    base_model = Sequential()
    base_model.add(Dense(input_dim=14, units=1))
    base_model.add(Dense(out_dim, input_dim=14, init='uniform', activation=activation))
    base_model.add(Dense(14, init='uniform', activation=activation))
    base_model.add(Dense(1, init='uniform', activation='sigmoid'))
    base_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return base_model

def multiperceptron(train_x, train_y, results):
    #train_x, test_x, train_y, test_y, indices_train, indices_test = train_test_split(load_source2, load_target2, load_source_phrase2, test_size=0.3)
    activation = ["relu", "sigmoid"]
    optimizer = ["adam", "adagrad"]
    out_dim = [20, 50, 100, 200]
    epochs = [20, 50]
    batch_size = [5, 10]

    # create model    
    model = KerasClassifier(build_fn=base_model, verbose=1)
    
    # grid search
    param_grid = dict(activation=activation, 
                optimizer=optimizer, 
                out_dim=out_dim, 
                epochs=epochs, 
                batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    
    # Fit the model
    grid_result = grid.fit(train_x, train_y[:, np.newaxis])
    print ("best_score:{0}, best_params:{1}".format(grid_result.best_score_, grid_result.best_params_))

    grid.model.save('./'+results+'/phrase_classifier.mm')
    #scores = model.evaluate(test_x, test_y)
    #print("\n")
    #print(model.metrics_names, scores)
    #predictions = np.round(model.predict(test_x))
    #correct = test_y[:, np.newaxis]
    #extracted correct phrases
    #print(indices_test[np.array(predictions == correct).flatten()])
    return grid

def classification(X_train, y_train, X_test, y_test, results):
    parameters = {
        'n_estimators'      : [10, 50, 100, 200, 300, 400, 500],
        'random_state'      : [0],
        'n_jobs'            : [200],
        'max_features'      : ['auto', 'log2', 'sqrt', None],
        'criterion'         : ['gini'],
        'max_depth'         : [3, 5, 10, 20, 30, 40, 50, 100]
    }
    #random forest
    #clf = make_pipeline(
    #    preprocessing.StandardScaler(),
    #    GridSearchCV(RandomForestClassifier(), parameters))
    #clf = svm.SVC(C=1.0, gamma=0.001)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    #Serialize
    joblib.dump(clf, './'+results+'/phrase.pkl')
    #clf = joblib.load('./'+results+'/phrase.pkl')

    return clf

def load_features(recalc=None, results=None):
    if recalc == 1:
        files = glob.glob(results+"/sick_train_*.err")
        files2 = glob.glob(results+"/sick_trial_*.err")
        files = files + files2
        target = []
        source = []
        source_phrase = []
        filenames = []
        for file in files:
            f = open(file,"r")
            filename = re.search("sick_([a-z]*_[0-9]*)\.", file).group(1)
            #print(filename)
            try:
                temp = {i : json.loads(line) for i, line in enumerate(f)}
                for t in range(0, len(temp)+1):
                    for k, v in temp[t]["features"].items():
                        score = temp[t]["validity"]
                        kind = k
                        feature = v
                        #print(score, kind, feature)
                        target.append(score)
                        source.append(feature)
                        source_phrase.append(kind)
                        filenames.append(filename)
            except:
                continue

        with open(results+'/features.pickle', 'wb') as out_f:
            np.save(out_f, target)
            np.save(out_f, source)
            np.save(out_f, source_phrase)
            np.save(out_f, filenames)
    else:
        with open(results+'/features.pickle', 'rb') as in_f:
            target = np.load(in_f)
            source = np.load(in_f)
            source_phrase = np.load(in_f)
            filenames = np.load(in_f)
    return target, source, source_phrase

def all_score(outputs, trial_targets, phrases, results):
    with open('./'+results+'/all_result_rte.txt', 'w') as out_f:
        out_f.write('phrase\tpred\tgold\n')
        for i, line in enumerate(outputs):
            out_f.write('{0}\t{1}\t{2}\n'.format(phrases[i], line, trial_targets[i]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    # Get training and trial features
    target, source, source_phrase = load_features(1, args.results)
    #random.seed(23)
    #random.shuffle(train)
    #random.shuffle(test)
    half = int(len(target)/2)
    train_sources, trial_sources = source[:half], source[half:]
    train_targets, trial_targets = target[:half], target[half:]
    train_phrases, trial_phrases = source_phrase[:half], source_phrase[half:]
    print ('test size: {0}, training size: {1}'.format(len(trial_targets), len(train_targets)))
    
    # Train the regressor
    #clf = classification(train_sources, train_targets, trial_sources, trial_targets, args.results)
    #Train multiperceptron with training dataset
    clf = multiperceptron(np.array(source), np.array(target), args.results)

    # Apply regressor to trial data
    #outputs = clf.predict(trial_sources)
    #all_score(outputs, trial_targets, trial_phrases, args.results)
    #trial_targets = list(map(str, trial_targets))
    #outputs = list(map(str, list(outputs)))
    #f = open('./'+args.results+'/report.txt', 'w')
    #f.write(classification_report(trial_targets, outputs, digits=4))
    #f.close()

    

if __name__ == '__main__':
    main()

#sick_trial_24
#sick_trial_816
#sick_trial_3941
#sick_trial_2972
#sick_test_9491
#sick_trial_4
#sick_test_408
#sick_test_96
#sick_test_3628
#sick_test_2367
#sick_train_8073
#sick_test_4479
#sick_trial_6634

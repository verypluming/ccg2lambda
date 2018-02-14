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
import pandas as pd
import argparse
import random
import difflib
from time import time
import itertools
import datetime
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing, linear_model, svm
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, train_test_split

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, Embedding, LSTM, Merge, Bidirectional, Concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

def crossvalidation(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    return scores.mean(), scores.std()

def text_to_word_list(text):
    #preprocess and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"_", " _ ", text)

    text = text.split()

    return text



def siamese_mlp_model(load_target, load_sick_id, premises, subgoals, results):
    # concatenate sentence vector obtained from siamese-model and premise-subgoal similarity vector obtained from proof
    # Prepare embedding
    EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
    stops = set(stopwords.words('english'))
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    sentences_cols = ['sentence1', 'sentence2', 'premise', 'subgoal']

    # Load training and test set
    inputfile_sentences = []
    for i, label in enumerate(load_target):
        f = open("plain/sick_"+load_sick_id[i]+".txt", "r")
        s = f.readlines()
        f.close()
        premise = premises[i]
        subgoal = subgoals[i]
        target = label
        inputfile_sentences.append([s[0].strip(), s[1], premise, subgoal, target])
    df = pd.DataFrame(inputfile_sentences, columns=['sentence1', 'sentence2', 'premise', 'subgoal', 'target'])
    half = int(len(inputfile_sentences)/2)
    train_df = df[:half]
    test_df = df[half:]

    # Iterate over the sentences only of both training and test datasets
    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():

            # Iterate through the text of both sentences of the row
            for sentence in sentences_cols:
                q2n = []  # q2n -> sentence numbers representation
                for word in text_to_word_list(row[sentence]):
                    # Check for unwanted words
                    if word in stops and word not in word2vec.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
                # Replace sentence as word to sentence as number representation
                dataset.set_value(index, sentence, q2n)
    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)

    del word2vec

    max_seq_length = max(train_df.sentence1.map(lambda x: len(x)).max(),
                     train_df.sentence2.map(lambda x: len(x)).max(),
                     test_df.sentence1.map(lambda x: len(x)).max(),
                     test_df.sentence2.map(lambda x: len(x)).max())
    
    max_phrase_length = max_seq_length #temporarily
    #max_phrase_length = max(train_df.premise.map(lambda x: len(x)).max(),
    #                 train_df.subgoal.map(lambda x: len(x)).max(),
    #                 test_df.premise.map(lambda x: len(x)).max(),
    #                 test_df.subgoal.map(lambda x: len(x)).max())
    # Split to train validation
    validation_size = 2000
    X = train_df[sentences_cols]
    Y = train_df['target']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.sentence1, 'right': X_train.sentence2, 'premise': X_train.premise, 'subgoal': X_train.subgoal}
    X_validation = {'left': X_validation.sentence1, 'right': X_validation.sentence2, 'premise': X_validation.premise, 'subgoal': X_validation.subgoal}
    X_test = {'left': test_df.sentence1, 'right': test_df.sentence2, 'premise': test_df.premise, 'subgoal': test_df.subgoal}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    for dataset, side in itertools.product([X_train, X_validation], ['premise', 'subgoal']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_phrase_length)

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 64
    n_epoch = 50

    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = Bidirectional(LSTM(n_hidden))

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    th_output = Concatenate()([left_output, right_output])

    premise_input = Input(shape=(max_phrase_length,), dtype='int32')
    subgoal_input = Input(shape=(max_phrase_length,), dtype='int32')
    
    phrase_embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_phrase_length, trainable=False)
    
    encoded_premise = phrase_embedding_layer(premise_input)
    encoded_subgoal = phrase_embedding_layer(subgoal_input)

    shared_phrase_lstm = Bidirectional(LSTM(n_hidden))

    premise_output = shared_phrase_lstm(encoded_premise)
    subgoal_output = shared_phrase_lstm(encoded_subgoal)

    ps_output = Concatenate()([premise_output, subgoal_output])

    merge_input = Concatenate()([th_output, ps_output])

    predictions = Dense(1, init='uniform', activation='sigmoid')(merge_input)
    model = Model([left_input, right_input, premise_input, subgoal_input], outputs=predictions)

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    plot_model(model, to_file='./'+results+'/siamese_mlp_model.png', show_shapes=True, show_layer_names=True)

    # Start training
    training_start_time = time()

    trained = model.fit([X_train['left'], X_train['right'], X_train['premise'], X_train['subgoal']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right'], X_validation['premise'], X_validation['subgoal']], Y_validation))

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
    trained.save('./'+results+'/siamese_mlp_model.mm')
    return trained

def siamese_mlp_model_before(load_source, load_target, load_sick_id, results):
    # concatenate sentence vector obtained from siamese-model and premise-subgoal similarity vector obtained from proof
    # Prepare embedding
    EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
    stops = set(stopwords.words('english'))
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    sentences_cols = ['sentence1', 'sentence2', 'sim']
    sentences_cols2 = ['sentence1', 'sentence2']

    # Load training and test set
    inputfile_sentences = []
    for i, label in enumerate(load_target):
        f = open("plain/sick_"+load_sick_id[i]+".txt", "r")
        s = f.readlines()
        f.close()
        sim = load_source[i]
        target = label
        inputfile_sentences.append([s[0].strip(), s[1], sim, target])
    df = pd.DataFrame(inputfile_sentences, columns=['sentence1', 'sentence2', 'sim', 'target'])
    half = int(len(inputfile_sentences)/2)
    train_df = df[:half]
    test_df = df[half:]

    # Iterate over the sentences only of both training and test datasets
    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():

            # Iterate through the text of both sentences of the row
            for sentence in sentences_cols:
                q2n = []  # q2n -> sentence numbers representation
                for word in text_to_word_list(row[sentence]):
                    # Check for unwanted words
                    if word in stops and word not in word2vec.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
                # Replace sentence as word to sentence as number representation
                dataset.set_value(index, sentence, q2n)
    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)

    del word2vec

    max_seq_length = max(train_df.sentence1.map(lambda x: len(x)).max(),
                     train_df.sentence2.map(lambda x: len(x)).max(),
                     test_df.sentence1.map(lambda x: len(x)).max(),
                     test_df.sentence2.map(lambda x: len(x)).max())

    # Split to train validation
    validation_size = 2000
    X = train_df[sentences_cols]
    Y = train_df['target']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.sentence1, 'right': X_train.sentence2, 'sim': X_train.sim}
    X_validation = {'left': X_validation.sentence1, 'right': X_validation.sentence2, 'sim': X_validation.sim}
    X_test = {'left': test_df.sentence1, 'right': test_df.sentence2, 'sim': test_df.sim}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
    X_train['sim'] = pad_sequences(X_train['sim'], maxlen=28)
    X_validation['sim'] = pad_sequences(X_validation['sim'], maxlen=28)

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 64
    n_epoch = 25

    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = Bidirectional(LSTM(n_hidden))

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    th_output = Concatenate()([left_output, right_output])

    sim_input = Input(shape=(28,))
    sim_input1 = Dense(100, activation='relu')(sim_input)
    sim_input2 = Dense(28, activation='relu')(sim_input1)
    merge_input = Concatenate()([th_output, sim_input2])

    predictions = Dense(1, init='uniform', activation='sigmoid')(merge_input)
    model = Model([left_input, right_input, sim_input], outputs=predictions)

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    plot_model(model, to_file='./'+results+'/siamese_mlp_model.png', show_shapes=True, show_layer_names=True)

    # Start training
    training_start_time = time()

    trained = model.fit([X_train['left'], X_train['right'], X_train['sim']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right'], X_validation['sim']], Y_validation))

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
    trained.save('./'+results+'/siamese_mlp_model.mm')
    return trained


def base_model(activation="relu", optimizer="adam", out_dim=20):
    base_model = Sequential()
    base_model.add(Dense(input_dim=28, units=1))
    base_model.add(Dense(out_dim, input_dim=28, init='uniform', activation=activation))
    base_model.add(Dense(28, init='uniform', activation=activation))
    base_model.add(Dense(1, init='uniform', activation='sigmoid'))
    base_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return base_model

def multiperceptron(train_x, train_y, results):
    #train_x, test_x, train_y, test_y, indices_train, indices_test = train_test_split(load_source2, load_target2, load_source_phrase2, test_size=0.3)
    activation = ["relu", "sigmoid"]
    optimizer = ["adam", "adagrad"]
    out_dim = [50, 100, 150, 200]
    epochs = [20, 50]
    batch_size = [5, 10]

    # grid search parameter
    param_grid = dict(mlp__activation=activation, 
                mlp__optimizer=optimizer, 
                mlp__out_dim=out_dim, 
                mlp__epochs=epochs, 
                mlp__batch_size=batch_size)

    # create pipeline
    estimators = []
    estimators.append(('mlp', KerasClassifier(build_fn=base_model, verbose=1)))
    pipeline = Pipeline(estimators)
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5)
    
    # fit model
    grid_result = grid.fit(train_x, train_y[:, np.newaxis])
    clf = grid_result.best_estimator_
    print(grid_result.best_estimator_)
    print ("best_score:{0}, best_params:{1}".format(grid_result.best_score_, grid_result.best_params_))

    # save model
    clf.steps[0][1].model.save('./'+results+'/phrase_classifier.mm')
    #scores = model.evaluate(test_x, test_y)
    #print("\n")
    #print(model.metrics_names, scores)
    #predictions = np.round(model.predict(test_x))
    #correct = test_y[:, np.newaxis]
    #extracted correct phrases
    #print(indices_test[np.array(predictions == correct).flatten()])
    return clf

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
        premises = []
        subgoals = []
        filenames = []
        for file in files:
            f = open(file,"r")
            filename = re.search("sick_([a-z]*_[0-9]*)\.", file).group(1)
            #print(filename)
            try:
                temp = {i : json.loads(line) for i, line in enumerate(f)}
                for t in range(0, len(temp)+1):
                    for k, v in temp[t]["phrases"].items():
                        score = temp[t]["validity"]
                        premise = k
                        subgoal = v
                        #print(score, kind, feature)
                        target.append(score)
                        premises.append(premise)
                        subgoals.append("".join(subgoal))
                        filenames.append(filename)
            except:
                continue

        with open(results+'/features.pickle', 'wb') as out_f:
            np.save(out_f, target)
            np.save(out_f, premises)
            np.save(out_f, subgoals)
            np.save(out_f, filenames)
    else:
        with open(results+'/features.pickle', 'rb') as in_f:
            target = np.load(in_f)
            premises = np.load(in_f)
            subgoals = np.load(in_f)
            filenames = np.load(in_f)
    return target, premises, subgoals, filenames

def load_features_before(recalc=None, results=None):
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
    return target, source, source_phrase, filenames

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
    #target, source, source_phrase, filenames = load_features(1, args.results)
    target, premises, subgoals, filenames = load_features(1, args.results)
    #random.seed(23)
    #random.shuffle(train)
    #random.shuffle(test)
    half = int(len(target)/2)
    #train_sources, trial_sources = source[:half], source[half:]
    #train_targets, trial_targets = target[:half], target[half:]
    #train_phrases, trial_phrases = source_phrase[:half], source_phrase[half:]
    train_targets, trial_targets = target[:half], target[half:]
    train_premises, trial_premises = premises[:half], premises[half:]
    train_subgoals, trial_subgoals = subgoals[:half], subgoals[half:]
    print ('test size: {0}, training size: {1}'.format(len(trial_targets), len(train_targets)))
    
    # Train the regressor
    #clf = classification(train_sources, train_targets, trial_sources, trial_targets, args.results)
    #Train multiperceptron with training dataset
    #clf = multiperceptron(np.array(source), np.array(target), args.results)
    siamese = siamese_mlp_model(np.array(target), np.array(filenames), np.array(premises), np.array(subgoals), args.results)

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
#sick_trial_3941 bug!!
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

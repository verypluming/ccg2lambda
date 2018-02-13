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

from __future__ import print_function
from collections import OrderedDict, defaultdict
import json
import logging
import re
from subprocess import Popen
import subprocess
import sys
from py_thesaurus import Thesaurus

from abduction_tools import *
from knowledge import get_tokens_from_xml_node, get_lexical_relations_from_preds
from normalization import denormalize_token, normalize_token
from nltk.corpus import wordnet as wn
import difflib
from scipy.spatial import distance
from semantic_tools import is_theorem_defined
from tactics import get_tactics
from tree_tools import is_string
from linguistic_tools import linguistic_relationship, get_wordnet_cascade
#from itertools import chain
import urllib.parse
import unicodedata
from keras.models import load_model
import numpy as np

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import itertools
import os
import pandas as pd
import scipy as sp
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Embedding, LSTM, Merge, Bidirectional, Concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from lxml import etree

model = load_model('./sick_results/siamese_mlp_model.mm')

class AxiomsPhrase(object):
    """
    Create phrasal axioms 
    """
    def __init__(self):
        pass

    def attempt(self, coq_scripts, doc, target, context=None):
        return TryPhraseAbduction(coq_scripts, target, doc)

def get_premise_lines_ex(coq_output_lines):
    premise_lines = []
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(
        coq_output_lines)
    if not line_index_last_conclusion_sep:
        return premise_lines
    for line in coq_output_lines[line_index_last_conclusion_sep - 1:0:-1]:
        if line == "":
            return premise_lines
        else:
            premise = ""
            if re.search("=", line) and not re.search("\(", line):
                # Subj x1 = x2, Acc ?3353 = x1
                premise= line
            if re.search("\s_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*$", line):
                # _pred x0
                premise = re.search("\s(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*)$", line).group(1)
            if re.search("\s_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s[xz\?][0-9]*$", line):
                # _pred x0 x1
                premise = re.search("\s(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s[xz\?][0-9]*)", line).group(1)
            if re.search("\s_[a-zA-Z]*_?[1-9]?\s\([a-zA-Z]*\s[xz\?][0-9]*\)", line):
                # _pred (Subj x0)
                premise = re.search("\s(_[a-zA-Z]*_?[1-9]?\s\([a-zA-Z]*\s[xz\?][0-9]*\))", line).group(1)
            if re.search("\s_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s\([a-zA-Z]*\s[xz\?][0-9]*\)", line):
                # _pred x (Subj x0)
                premise = re.search("\s(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s\([a-zA-Z]*\s[xz\?][0-9]*\))", line).group(1)
            if premise != "":
                #print("premise:{0}".format(premise), file=sys.stderr)
                premise_lines.append(premise)
    return premise_lines

def get_conclusion_lines_ex(coq_output_lines):
    conclusion_lines = []
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(coq_output_lines)
    if not line_index_last_conclusion_sep:
        return None
    #print(coq_output_lines[line_index_last_conclusion_sep+1:])
    for line in coq_output_lines[line_index_last_conclusion_sep+1:]:
        if re.search('Toplevel', line):
            return conclusion_lines
        elif line == '':
            continue
        elif re.search("No more subgoals", line):
            conclusion_lines.append(line)
        elif re.search("subgoal", line):
            continue
        elif re.search('repeat nltac_base', line):
            return conclusion_lines
        else:
            conclusion = ""
            if re.search("=", line) and not re.search("\(", line):
                # Subj x1 = x2, Acc ?3353 = x1
                conclusion = line
            if re.search("False", line):
                # False
                conclusion = "False"
            if re.search("_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*$", line):
                # _pred(_1) x0
                conclusion = re.search("(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*)$", line).group(1)
            if re.search("_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s[xz\?][0-9]*$", line):
                # _pred x0 x1
                conclusion = re.search("(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s[xz\?][0-9]*)", line).group(1)
            if re.search("_[a-zA-Z]*_?[1-9]?\s\([a-zA-Z]*\s[xz\?][0-9]*\)", line):
                # _pred (Subj x0)
                conclusion = re.search("(_[a-zA-Z]*_?[1-9]?\s\([a-zA-Z]*\s[xz\?][0-9]*\))", line).group(1)
            if re.search("_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s\([a-zA-Z]*\s[xz\?][0-9]*\)", line):
                # _pred x (Subj x0)
                conclusion = re.search("(_[a-zA-Z]*_?[1-9]?\s[xz\?][0-9]*\s\([a-zA-Z]*\s[xz\?][0-9]*\))", line).group(1)
            if conclusion != "":
                #print("conclusion:{0}".format(conclusion), file=sys.stderr)
                conclusion_lines.append(conclusion)
    return conclusion_lines


def TryPhraseAbduction(coq_scripts, target, doc):
    #assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    features = {}
    direct_proof_scripts, reverse_proof_scripts = [], []
    inference_result_str, all_scripts = "unknown", []

    #entailment proof
    inference_result_str, direct_proof_scripts, new_direct_axioms, features = \
        try_phrase_abduction(direct_proof_script, doc, 
                            previous_axioms=axioms, features=features, expected='yes', target=target)
    current_axioms = axioms.union(new_direct_axioms)
    current_features = features
    if not inference_result_str == 'yes':
        #contradiction proof
        inference_result_str, reverse_proof_scripts, new_reverse_axioms, features = \
            try_phrase_abduction(reverse_proof_script, doc, 
                            previous_axioms=current_axioms, features=current_features, expected='no', target=target)
        current_axioms.update(new_reverse_axioms)
    all_scripts = direct_proof_scripts + reverse_proof_scripts
    axioms = current_axioms
    return inference_result_str, all_scripts
    
def try_phrase_abduction(coq_script, doc, previous_axioms=set(), features={}, expected='yes', target='yes'):
    current_features = features
    new_coq_script = insert_axioms_in_coq_script(previous_axioms, coq_script)
    current_tactics = get_tactics()
    #debug_tactics = 'repeat nltac_base. try substitution. Qed'
    debug_tactics = 'repeat nltac_base. Qed'
    coq_script_debug = new_coq_script.replace(current_tactics, debug_tactics)
    process = Popen(
        coq_script_debug,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    premise_lines = get_premise_lines_ex(output_lines)
    conclusion = get_conclusion_lines_ex(output_lines)
    if is_theorem_almost_defined(output_lines):
        if expected == target and target != "unknown":
            #positive label
            features_log = {"phrases": current_features, "validity": 1.0, "gold": target, "proof": expected, "premise": premise_lines, "subgoals": conclusion}
            print(json.dumps(features_log), file=sys.stderr)
        elif target == "unknown" and expected != "unknown":
            #negative label
            features_log = {"phrases": current_features, "validity": 0.0, "gold": target, "proof": expected, "premise": premise_lines, "subgoals": conclusion}
            print(json.dumps(features_log), file=sys.stderr)
        return expected, [new_coq_script], previous_axioms, features
    #premise_lines = get_premise_lines(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    #conclusion = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusion:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms, features
    axioms, features = make_phrase_axioms(premise_lines, conclusion, doc, output_lines, expected, coq_script_debug)
    current_features.update(features)
    #axioms = filter_wrong_axioms(axioms, coq_script) temporarily
    #add only newly generated axioms
    axioms = axioms.difference(previous_axioms)
    new_coq_script = insert_axioms_in_coq_script(axioms, coq_script_debug)
    process = Popen(
        new_coq_script,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    inference_result_str = expected if is_theorem_almost_defined(output_lines) else 'unknown'
    if inference_result_str == target and target != "unknown":
        #positive label
        features_log = {"phrases": current_features, "validity": 1.0, "gold": target, "proof": expected, "premise": premise_lines, "subgoals": conclusion}
        print(json.dumps(features_log), file=sys.stderr)
    elif target == "unknown" and inference_result_str != "unknown":
        #negative label
        features_log = {"phrases": current_features, "validity": 0.0, "gold": target, "proof": expected, "premise": premise_lines, "subgoals": conclusion}
        print(json.dumps(features_log), file=sys.stderr)
    return inference_result_str, [new_coq_script], axioms, features

def make_phrase_axioms(premises, conclusions, doc, coq_output_lines=None, expected='yes', coq_script_debug=None):
    axioms = set()
    features = {}
    #check sub-goals with normal variables
    #conclusions_normal = distinguish_normal_conclusions(conclusions)

    #if existential variables contain in sub-goals, create axioms for sub-goals with existential variables at first
    axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions, doc, coq_script_debug, expected)

    #create axioms for sub-goals with normal variables
    #for conclusion in conclusions_normal:
    #    matching_premises = get_premises_that_partially_match_conclusion_args(premises, conclusion)
    #    premise_preds = [premise.split()[2] for premise in matching_premises]
    #    pred_args = get_predicate_case_arguments(matching_premises, conclusion)
    #    axioms.update(make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion, pred_args, expected))
    #    if not axioms:
    #        failure_log = make_failure_log(
    #            conclusion, premise_preds, conclusion, premises, coq_output_lines)
    #        print(json.dumps(failure_log), file=sys.stderr)
    return axioms, features

def get_conclusion_lines(coq_output_lines):
    conclusion_lines = []
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(coq_output_lines)
    if not line_index_last_conclusion_sep:
        return None
    for line in coq_output_lines[line_index_last_conclusion_sep+1:]:
        if re.search('Toplevel', line):
            return conclusion_lines
        elif line == '':
            continue
        elif re.search("No more subgoals", line):
            conclusion_lines.append(line)
        elif re.search("subgoal", line):
            continue
        elif re.search('repeat nltac_base', line):
            return conclusion_lines
        else:
            conclusion_lines.append(line)
    return conclusion_lines

def check_case_from_list(total_arg_list):
    new_total_arg_list = []
    for t in total_arg_list:
        if contains_case(t):
            t_arg = re.search("([xyz][0-9]*)", t).group(1)
            new_total_arg_list.append(t_arg)
        else:
            new_total_arg_list.append(t)
    sorted_new_total_arg_list = list(set(new_total_arg_list))
    return sorted_new_total_arg_list

def is_theorem_almost_defined(output_lines):
    #check if all content subgoals are deleted(remaining relation subgoals can be permitted)
    #ignore relaional subgoals(False, Acc x0=x1) in the proof
    conclusions = []
    conclusions = get_conclusion_lines_ex(output_lines)
    #print("conclusion:{0}".format(conclusions), file=sys.stderr)
    subgoalflg = 0
    if conclusions is None:
        return False
    if len(conclusions) > 0:
        for conclusion in conclusions:
            #if not "False" in conclusion:
            if not "=" in conclusion:
                subgoalflg = 1
            if "No more subgoals" in conclusion:
                return True
    if subgoalflg == 1:
        return False
    else:
        return True

def get_tree_pred_args_ex(line, is_conclusion=False):
    """
    Given the string representation of a premise, where each premise is:
      pX : predicate1 (arg1 arg2 arg3)
      pY : predicate2 arg1
    or the conclusion, which is of the form:
      predicate3 (arg2 arg4)
    returns the list  of variables (tree leaves).
    """
    tree_args = None
    #if not is_conclusion:
    #    line = ' '.join(line.split()[2:])
    # Transform a line 'Subj ?2914 = Acc x1' into '= (Subj ?2914) (Acc x1)'
    line = re.sub(r'(.+) (.+) = (.+) (.+)', r'= (\1 \2) (\3 \4)', line)
    # Transform a line '?2914 = Acc x1' into '= ?2914 (Acc x1)'
    line = re.sub(r'(.+) = (.+) (.+)', r'= \1 (\2 \3)', line)
    # Transform a line 'Subj ?2914 = x1' into '= (Subj ?2914) x1'
    line = re.sub(r'(.+) (.+) = (.+)', r'= (\1 \2) \3', line)
    # Transform a line 'Subj ?2914 = x1' into '= (Subj ?2914) x1'
    line = re.sub(r'(.+) = (.+)', r'= \1 \2', line)
    tree_args = parse_coq_line(line)
    if tree_args is None or is_string(tree_args) or len(tree_args) < 1:
        return None
    return [str(child) for child in tree_args if str(child) != '=']
    # return list(set([str(child) for child in tree_args if str(child) != '='] + tree_args.leaves()))

def contains_case(coq_line):
    """
    Returns True if the coq_line contains a case predicate, e.g.
    'H0 : _meat (Acc x1)'
    'H : _lady (Subj x1)'
    Returns False otherwise.
    We assume that case is specified by an uppercase character
    followed by at least two lowercased characters, e.g. Acc, Subj, Dat, etc.
    """
    if re.search(r'[A-Z][a-z][a-z]', coq_line):
        return True
    return False

def get_pred_from_coq_line(line, is_conclusion=False):
    # Transform a line 'Subj ?2914 = Acc x1' into '= (Subj ?2914) (Acc x1)'
    line = re.sub(r'(.+) (.+) = (.+) (.+)', r'= (\1 \2) (\3 \4)', line)
    # Transform a line '?2914 = Acc x1' into '= ?2914 (Acc x1)'
    line = re.sub(r'(.+) = (.+) (.+)', r'= \1 (\2 \3)', line)
    # Transform a line 'Subj ?2914 = x1' into '= (Subj ?2914) x1'
    line = re.sub(r'(.+) (.+) = (.+)', r'= (\1 \2) \3', line)
    # Transform a line 'Subj ?2914 = x1' into '= (Subj ?2914) x1'
    line = re.sub(r'(.+) = (.+)', r'= \1 \2', line)
    if is_conclusion:
        return line.split()[0]
    else:
        return line.split()[0]
        #if len(line.split()) > 2:
        #    return line.split()[2]
        #else:
        #    return line.split()[0]
    raise(ValueError("Strange coq line: {0}".format(line)))

def check_decomposed(line):
    if line.startswith(':'):
        return False
    if re.search("forall", line):
        return False
    if re.search("exists", line):
        return False
    return True

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

def embedding_sentence(sentence):
    # to do: embeddings matrix and vocabulary should be created and saved
    # to do: check if the format of sentence vector is correct(not using dataframe)
    from gensim import corpora

    # Prepare embedding
    # to do: to be faster, load EMBEDDING_FILE using API https://github.com/RaRe-Technologies/gensim-data
    EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
    stops = set(stopwords.words('english'))
    #vocabulary = dict()
    g = open('./sick_vocab.txt', 'r')
    vocabulary = json.load(g)
    inverse_vocabulary = ['<unk>'] # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    q2n = []  # q2n -> sentence numbers representation
    max_seq_length = 26 # check max_seq_length
    for word in text_to_word_list(sentence):
        # Check for unwanted words
        if word in stops and word not in word2vec.vocab:
            continue
        # Just in case
        #if word not in vocabulary:
        #    vocabulary[word] = len(inverse_vocabulary)
        #    q2n.append(len(inverse_vocabulary))
        #    inverse_vocabulary.append(word)
        #else:
        q2n.append(vocabulary[word])
    sentence_vector = pad_sequences([q2n], maxlen=max_seq_length)
    return sentence_vector

def get_sentence_vectors(doc):
    sentence_vectors = []
    tokens = doc.xpath('//tokens')
    for i in range(len(tokens)):
        sentence_surface = ' '.join(tokens[i].xpath('token/@surf'))
        sentence_vector = embedding_sentence(sentence_surface)
        sentence_vectors.append(sentence_vector)
    return np.array(sentence_vectors)

def make_phrases_from_premises_and_conclusions_ex(premises, conclusions, doc, coq_script_debug=None, expected="yes"):
    sentence_vectors = get_sentence_vectors(doc) # to do: confirm if doc is correct
    t_vector = sentence_vectors[0].reshape(1,26)
    h_vector = sentence_vectors[1].reshape(1,26)
    coq_lists, param_lists, type_lists = [], [], []
    if coq_script_debug:
        coq_lists = coq_script_debug.split("\n")
        param_lists = [re.sub("Parameter ", "", coq_list) for coq_list in coq_lists if re.search("Parameter", coq_list)]
        type_lists = [param_list.split(":") for param_list in param_lists]
    features = defaultdict(list)
    covered_conclusions = set()
    used_premises = set()
    axioms = []
    phrase_pairs = []
    premises = [p for p in premises if get_pred_from_coq_line(p).startswith('_') and check_decomposed(p)]
    args_id = defaultdict(lambda: len(args_id))
    p_pred_args_id = {}
    c_pred_args_id = {}

    p_pred_args = {}
    #p_pred_args = defaultdict(list)
    for p in premises:
        predicate = get_pred_from_coq_line(p, is_conclusion=False)
        args = get_tree_pred_args_ex(p, is_conclusion=False)
        if args is not None:
            p_pred_args[predicate] = args
            #p_pred_args[predicate].append(args) # List of lists of args.
            p_pred_args_id[predicate] = [args_id[arg] for arg in args]

    # Compute relations between arguments as frozensets.
    #p_args_preds = defaultdict(set)
    #for pred, args_list in p_pred_args.items():
    #    for args in args_list:
    #        for arg in args:
    #            p_args_preds[frozenset([arg])].add(pred)
    #        p_args_preds[frozenset(args)].add(pred)
    # from pudb import set_trace; set_trace()
    #for args, preds in sorted(p_args_preds.items(), key=lambda x: len(x[0])):
    #    for targs, _ in sorted(p_args_preds.items(), key=lambda x: len(x[0])):
    #        # if args.intersection(targs):
    #        # E.g. if args is {'?4844'} and targs is {'(Acc ?4844)', 'x1'},
    #        # then we merge the predicates associated to these arguments.
    #        if any(a in ta for a in args for ta in targs):
    #            p_args_preds[targs].update(preds)

    c_pred_args = defaultdict(list)
    for c in conclusions:
        predicate = get_pred_from_coq_line(c, is_conclusion=True)
        args = get_tree_pred_args_ex(c, is_conclusion=True)
        if args is not None:
            c_pred_args[predicate].append(args) # List of lists of args.
            c_pred_args_id[predicate] = [args_id[arg] for arg in args]

    # Compute relations between arguments as frozensets.
    c_args_preds = defaultdict(set)
    for pred, args_list in c_pred_args.items():
        for args in args_list:
            for arg in args:
                c_args_preds[frozenset([arg])].add(pred)
            c_args_preds[frozenset(args)].add(pred)
    # from pudb import set_trace; set_trace()
    for args, preds in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
        for targs, _ in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
            # if args.intersection(targs):
            # E.g. if args is {'?4844'} and targs is {'(Acc ?4844)', 'x1'},
            # then we merge the predicates associated to these arguments.
            if any(a in ta for a in args for ta in targs):
                c_args_preds[targs].update(preds)
    #create axioms about sub-goals containing case information
    case_c_preds = [c for c, c_args in c_pred_args.items() if contains_case(str(c_args)) and len(c_args[0]) == 1]
    for case_c_pred in case_c_preds:
        #new_instance = Thesaurus(re.sub("_", "", case_c_pred))
        #synonyms = new_instance.get_synonym()
        #antonyms = new_instance.get_antonym()
        case_c_arg = c_pred_args[case_c_pred][0][0]
        #gen_c_arg = re.sub(r'\([A-Z][a-z][a-z][a-z]?\s([xz][0-9])*\)',r'\1', case_c_arg) #extract x1 from Subj x1
        #ex_p_pred = [prem for prem, p_args in p_pred_args.items() if set(p_args).issubset([gen_c_arg])] #x1 event
        #all_p_pred = [prem for prem, p_args in p_pred_args.items() if set(p_args).issuperset([gen_c_arg])] #x1 all predicates 
        #key_p_pred = set(all_p_pred).difference(set(ex_p_pred)) #all predicates - event
        #check_args = list(chain.from_iterable([p_pred_args[k] for k in key_p_pred])) #variables of key_p_pred
        #if gen_c_arg in check_args:
        #    check_args.remove(gen_c_arg) #variables of key_p_pred - gen_c_arg

        case = re.search(r'([A-Z][a-z][a-z][a-z]?)', case_c_arg).group(1)
        pat = re.compile(case)
        case_p_preds = []
        case_p_preds_args_id = []
        for p_pred, p_args in p_pred_args.items():
            if re.search(pat, p_args[0]) and case_c_arg in p_args:
                #if variable is not existential, predicates whose variables are the same as sub-goals are premise candidates
                case_p_preds.append(p_pred)
                case_p_preds_args_id.extend(p_pred_args_id[p_pred])
                continue
            elif re.search(pat, p_args[0]) and re.search("\?", case_c_arg):
                #if variable is existential, any predicates whose variables have the same semantic role are premise candidates
                case_p_preds.append(p_pred)
                case_p_preds_args_id.extend(p_pred_args_id[p_pred])
                continue
            #if set(p_args).issuperset(check_args):
            #    #NP-phrase
            #    case_p_preds.append(p_pred)
            #    case_p_preds_args_id.extend(p_pred_args_id[p_pred])
            #    continue
        if len(case_p_preds) > 0:
            features["".join(case_p_preds)].append(case_c_pred) #temporarily
            relation = ""
            axiom = 'Axiom ax_case_phrase{0}{1} : forall {2}, '.format(
                "".join(case_p_preds),
                case_c_pred,
                ' '.join('x' + str(i) for i in list(set(case_p_preds_args_id+c_pred_args_id[case_c_pred]))),
                )
            for case_p_pred in case_p_preds:
                #create axioms with premises which have the same case 
                axiom += case_p_pred+" "+" ".join('x' + str(i) for i in p_pred_args_id[case_p_pred])+" -> "
                #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                c_ph_word = re.sub("_", "", case_c_pred)
                p_ph_word = re.sub("_", "", case_p_pred)
                #typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                #ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                #wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                #word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                #simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                #check the predicate relation if their type is the same
                #if typesim == 1.0:
                #    if p_ph_word in synonyms:
                #       relation = "synonym"
                #    if p_ph_word in antonyms:
                #        relation = "antonym"
                #    if not relation:
                #        wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                #    if not relation:
                #        relation = check_stem(c_ph_word, p_ph_word)
                #    if not relation:
                #        relation = check_word2vec(c_ph_word, p_ph_word)
                #rte = check_rte(expected)
                #features[case_p_pred+case_c_pred] = simlist
                used_premises.add(case_p_pred)
                if relation == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for event predicates
                    axiom = 'Axiom ax_case_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                    break
                elif relation == "antonym" and "Entity -> Prop" in check_types(c_ph_word, type_lists):
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for entity->prop predicates
                    axiom = 'Axiom ax_case_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F x -> F y  -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                    break
                elif relation == "antonym":
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for entity predicates
                    axiom = 'Axiom ax_case_phrase_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                    break
            if relation != "antonym":
                axiom += case_c_pred+" "+' '.join('x' + str(i) for i in c_pred_args_id[case_c_pred])+"."
        #if relation:
        #    covered_conclusions.add(case_c_pred)
        #    axioms.append(axiom)
            covered_conclusions.add(case_c_pred)
            axioms.append(axiom)

    #create axioms about sub-goals without case information
    exclude_preds_in_conclusion = {
        get_pred_from_coq_line(l, is_conclusion=True) \
            for l in conclusions if contains_case(l) and not l.startswith('_')}

    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 0:
            premise_preds = sorted([
            #     prem for prem, p_args in p_pred_args.items() if set(p_args).issuperset(args) and not contains_case(str(p_args))])
                prem for prem, p_args in p_pred_args.items() if set(p_args).issubset(args) and not contains_case(str(p_args))])
            #print("premise_preds: {0}, c_preds: {1}, p_pred_args: {2}, args: {3}, c_args_preds: {4}".format(premise_preds, c_preds, p_pred_args, args, c_args_preds))
            if premise_preds:
                premise_preds_args_id = []
                synonyms = []
                antonyms = []
                relation = ""
                for p in c_preds:
                    c_ph_word = re.sub("_", "", p)
                    #new_instance = Thesaurus(c_ph_word)
                    #synonyms = new_instance.get_synonym()
                    #antonyms = new_instance.get_antonym()
                    for premise_pred in premise_preds:
                        #if the premise has been already used for axiom containing cases(ex. lady(Subj x) -> woman(Subj x), it will be unnecessary)
                        #if premise_pred in used_premises:
                        #    continue
                        premise_preds_args_id.extend(p_pred_args_id[premise_pred])
                        p_ph_word = re.sub("_", "", premise_pred)
                        wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                        #if calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists)) == 1.0:
                        #    if p_ph_word in synonyms:
                        #        # There is synonym with sub-goals. This shows phrase-axioms can be created.
                        #        relation = "synonym"
                        #    if p_ph_word in antonyms:
                        #        # There is synonym with sub-goals. This shows phrase-axioms can be created.
                        #        relation = "antonym"
                        #    if not relation:
                        #        wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                        #    if not relation:
                        #        relation = check_stem(c_ph_word, p_ph_word)
                        #    if not relation:
                        #        relation = check_word2vec(c_ph_word, p_ph_word)
                #if relation:
                #    phrase_pairs.append((premise_preds, c_preds, relation)) # Saved phrase pairs
                phrase_pairs.append((premise_preds, c_preds, relation)) # Saved phrase pairs

                for phrase_pair in phrase_pairs:                    
                    for p in phrase_pair[1]:
                        if p not in covered_conclusions:
                            features["".join(premise_preds)].append(p) #temporarily
                            axiom = 'Axiom ax_phrase_{0}{1}{2} : forall {3}, '.format(
                                phrase_pair[2],
                                "".join(premise_preds),
                                p,
                                ' '.join('x' + str(i) for i in list(set(premise_preds_args_id+c_pred_args_id[p])))
                                )
                            for premise_pred in phrase_pair[0]:
                                #if premise_pred in used_premises:
                                #    continue
                                axiom += premise_pred+" "+" ".join('x' + str(i) for i in p_pred_args_id[premise_pred])+" -> "
                                #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                                c_ph_word = re.sub("_", "", p)
                                p_ph_word = re.sub("_", "", premise_pred)
                                #typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                                #ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                                #wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                                #word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                                #simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                                #rte = check_rte(expected)
                                #features[premise_pred+p] = simlist
                                #TO DO: how to consider antonym phrasal axiom
                                if phrase_pair[2] == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                                    #check if entailment axiom was generated
                                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                                    if len(exist_axioms) > 0:
                                        for exist_axiom in exist_axioms:
                                            axioms.remove(exist_axiom)
                                    #antonym axiom for event predicates
                                    axiom = 'Axiom ax_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                                        premise_pred,
                                        p)
                                    break
                                elif phrase_pair[2] == "antonym" and "Entity -> Prop" in check_types(c_ph_word, type_lists):
                                    #check if entailment axiom was generated
                                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                                    if len(exist_axioms) > 0:
                                        for exist_axiom in exist_axioms:
                                            axioms.remove(exist_axiom)
                                    #antonym axiom for entity->prop predicates
                                    axiom = 'Axiom ax_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F x -> F y  -> False.'.format(
                                        premise_pred,
                                        p)
                                    break
                                elif phrase_pair[2] == "antonym":
                                    #check if entailment axiom was generated
                                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                                    if len(exist_axioms) > 0:
                                        for exist_axiom in exist_axioms:
                                            axioms.remove(exist_axiom)
                                    #antonym axiom for entity predicates
                                    axiom = 'Axiom ax_phrase_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                                        premise_pred,
                                        p)
                                    break
                            if phrase_pair[2] != "antonym":
                                axiom += p+" "+' '.join('x' + str(i) for i in c_pred_args_id[p])+"."
                            #print(axiom)
                            covered_conclusions.add(p)
                            axioms.append(axiom)

                        
                    

    #select premise features whose similarity are max and min
    #sum_dist = {}
    #feature_dist = {}
    #return_feature = {}
    #for k, v in features.items():
    #    keys = k.split("_")
    #    relation, premise, subgoal = keys[0], keys[1], keys[2]
    #    if subgoal in sum_dist:
    #        sum_dist[subgoal][premise] = sum(v)
    #        feature_dist[subgoal][premise] = v
    #    else:
    #        sum_dist[subgoal] = {premise: sum(v)}
    #        feature_dist[subgoal] = {premise: v}

    #for s, f in sum_dist.items():
    #    max_premise = max(f.items(), key=lambda x:x[1])[0]
    #    min_premise = min(f.items(), key=lambda x:x[1])[0]
    #    max_feature = feature_dist[s][max_premise]
    #    min_feature = feature_dist[s][min_premise]
    #    return_feature[max_premise+"_"+min_premise+"_"+s] = max_feature + min_feature
    #print(phrase_pairs) # this is a list of tuples of lists.

    #return set(axioms), return_feature
    new_axioms = []
    for axiom in axioms:
        #to do: consider how to create premise and sub-goal vector
        predict = np.round(model.predict([t_vector, h_vector, embedding_sentence(premise_vector), embedding_sentence(sub-goal_vector)]))
        #if axiom classifier is 1, confirm axioms
        if int(predict) == 1:
            new_axioms.append(axiom)
    return set(new_axioms), features

def make_phrases_from_premises_and_conclusions_ex_before(premises, conclusions, coq_script_debug=None, expected="yes"):
    coq_lists, param_lists, type_lists = [], [], []
    if coq_script_debug:
        coq_lists = coq_script_debug.split("\n")
        param_lists = [re.sub("Parameter ", "", coq_list) for coq_list in coq_lists if re.search("Parameter", coq_list)]
        type_lists = [param_list.split(":") for param_list in param_lists]
    features = {}
    covered_conclusions = set()
    used_premises = set()
    axioms = []
    phrase_pairs = []
    premises = [p for p in premises if get_pred_from_coq_line(p).startswith('_') and check_decomposed(p)]

    p_pred_args = {}
    for p in premises:
        predicate = get_pred_from_coq_line(p, is_conclusion=False)
        args = get_tree_pred_args_ex(p, is_conclusion=False)
        if args is not None:
            p_pred_args[predicate] = args

    c_pred_args = defaultdict(list)
    for c in conclusions:
        predicate = get_pred_from_coq_line(c, is_conclusion=True)
        args = get_tree_pred_args_ex(c, is_conclusion=True)
        if args is not None:
            c_pred_args[predicate].append(args) # List of lists of args.

    # Compute relations between arguments as frozensets.
    c_args_preds = defaultdict(set)
    for pred, args_list in c_pred_args.items():
        for args in args_list:
            for arg in args:
                c_args_preds[frozenset([arg])].add(pred)
            c_args_preds[frozenset(args)].add(pred)
    # from pudb import set_trace; set_trace()
    for args, preds in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
        for targs, _ in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
            # if args.intersection(targs):
            # E.g. if args is {'?4844'} and targs is {'(Acc ?4844)', 'x1'},
            # then we merge the predicates associated to these arguments.
            if any(a in ta for a in args for ta in targs):
                c_args_preds[targs].update(preds)

    #create axioms about sub-goals with existential variables containing case information
    case_c_preds = [c for c, c_args in c_pred_args.items() if contains_case(str(c_args)) and len(c_args[0]) == 1]
    for case_c_pred in case_c_preds:
        case_c_arg = c_pred_args[case_c_pred][0][0]
        case_c_arg = re.sub(r'\?([0-9]+)', r'x\1', case_c_arg)
        case = re.search(r'([A-Z][a-z][a-z][a-z]?)', case_c_arg).group(1)
        pat = re.compile(case)
        for p_pred, p_args in p_pred_args.items():
            if re.search(pat, p_args[0]):
                #create axioms with premises which have the same case 
                axiom = 'Axiom ax_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                        p_pred,
                        case_c_pred,
                        "x0",
                        "y0")
                #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                c_ph_word = re.sub("_", "", case_c_pred)
                p_ph_word = re.sub("_", "", p_pred)
                typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                #rte = check_rte(expected)
                feature = simlist + wordnetrel
                if relation == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for event predicates
                    axiom = 'Axiom ax_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        p_pred,
                        case_c_pred)
                elif relation == "antonym":
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for entity predicates
                    axiom = 'Axiom ax_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                        p_pred,
                        case_c_pred)

                used_premises.add(p_pred)
                covered_conclusions.add(case_c_pred)
                axioms.append(axiom)
                features[p_pred+case_c_pred] = feature

    #create axioms about sub-goals without case information
    exclude_preds_in_conclusion = {
        get_pred_from_coq_line(l, is_conclusion=True) \
            for l in conclusions if not l.startswith('_') and contains_case(l)}

    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 0:
            premise_preds = [
                p for p, p_args in p_pred_args.items() if set(p_args).issubset(args)]
            premise_preds = sorted([p for p in premise_preds if not contains_case(p)])
            if premise_preds:
                phrase_pairs.append((premise_preds, c_preds)) # Saved phrase pairs for Yanaka-san.
                for premise_pred in premise_preds:
                    #if the premise has been already used for axiom containing cases(ex. lady(Subj x) -> woman(Subj x), it will be unnecessary)
                    if premise_pred in used_premises:
                        continue
                    #premise_pred = premise_preds[0] #not only the first premise, but all premises are selected
                    for p in c_preds:
                        if p not in covered_conclusions:
                            c_num_args = max(len(cargs) for cargs in c_pred_args[p])
                            p_num_args = len(p_pred_args[premise_pred])
                            #print(premise_pred, p, p_pred_args[premise_pred], c_pred_args[p])
                            axiom = 'Axiom ax_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                                premise_pred,
                                p,
                                ' '.join('x' + str(i) for i in range(p_num_args)),
                                ' '.join('y' + str(i) for i in range(c_num_args)))

                            #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                            c_ph_word = re.sub("_", "", p)
                            p_ph_word = re.sub("_", "", premise_pred)
                            typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                            ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                            wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                            word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                            simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                            wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
                            #rte = check_rte(expected)
                            feature = simlist + wordnetrel
                            if relation == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                                #check if entailment axiom was generated
                                exist_axioms = [axiom for axiom in axioms if p in axiom]
                                if len(exist_axioms) > 0:
                                    for exist_axiom in exist_axioms:
                                        axioms.remove(exist_axiom)
                                #antonym axiom for event predicates
                                axiom = 'Axiom ax_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                                    premise_pred,
                                    p)
                                covered_conclusions.add(p)
                            elif relation == "antonym":
                                #check if entailment axiom was generated
                                exist_axioms = [axiom for axiom in axioms if p in axiom]
                                if len(exist_axioms) > 0:
                                    for exist_axiom in exist_axioms:
                                        axioms.remove(exist_axiom)
                                #antonym axiom for entity predicates
                                axiom = 'Axiom ax_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                                    premise_pred,
                                    p)
                                covered_conclusions.add(p)
                        #print(axiom, feature)
                        axioms.append(axiom)
                        features[premise_pred+p] = feature

    #select premise features whose similarity are max and min
    sum_dist = {}
    feature_dist = {}
    return_feature = {}
    for k, v in features.items():
        keys = k.split("_")
        relation, premise, subgoal = keys[0], keys[1], keys[2]
        if subgoal in sum_dist:
            sum_dist[subgoal][premise] = sum(v)
            feature_dist[subgoal][premise] = v
        else:
            sum_dist[subgoal] = {premise: sum(v)}
            feature_dist[subgoal] = {premise: v}

    for s, f in sum_dist.items():
        max_premise = max(f.items(), key=lambda x:x[1])[0]
        min_premise = min(f.items(), key=lambda x:x[1])[0]
        max_feature = feature_dist[s][max_premise]
        min_feature = feature_dist[s][min_premise]
        return_feature[max_premise+"_"+min_premise+"_"+s] = max_feature + min_feature
    #print(phrase_pairs) # this is a list of tuples of lists.

    return set(axioms), return_feature

def check_types(pred, type_lists):
    for type_list in type_lists:
        if pred in type_list[0]:
            return type_list[1]

def calc_typesim(c_type, p_type):
    if c_type == p_type:
        return 1.0
    else:
        return 0.0

def calc_word2vecsim(sub_pred, prem_pred):
    process = Popen(\
    'curl http://localhost:5000/word2vec/similarity?w1='+ sub_pred +'\&w2='+ prem_pred, \
    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    similarity, err = process.communicate()
    try:
        return float(similarity.decode())
    except ValueError:
        return 0.0

def check_wordnetrel(sub_pred, prem_pred):
    antonym = "phrase"
    rel_list = ['copy', 'inflection', 'derivation', 'synonym', 'antonym', 'hypernym', 'hyponym', 'sister', 'cousin', 'similar']
    relations = linguistic_relationship(prem_pred, sub_pred)
    relation = get_wordnet_cascade(relations)
    rel_vec = [1.0 if rel == relation else 0.0 for rel in rel_list]
    return rel_vec, relation

def check_stem(sub_pred, prem_pred):
    from nltk import stem
    stemmer = stem.LancasterStemmer()
    if stemmer.stem(sub_pred) == stemmer.stem(prem_pred):
        return "stem"
    elif linguistic_relationship(stemmer.stem(prem_pred), stemmer.stem(sub_pred)):
        return "wordnet_stem"
    else:
        porter = stem.PorterStemmer()
        if porter.stem(sub_pred) == porter.stem(prem_pred):
            return "stem"
        elif linguistic_relationship(porter.stem(prem_pred), porter.stem(sub_pred)):
            return "wordnet_stem"
        else:
            snow = stem.SnowballStemmer("english")
            if snow.stem(sub_pred) == snow.stem(prem_pred):
                return "stem"
            elif linguistic_relationship(snow.stem(prem_pred), snow.stem(sub_pred)):
                return "wordnet_stem"
            else:
                return ""

def check_rte(expected):
    rte_list = ['yes', 'no', 'unknown']
    rte_vec = [1.0 if rte == expected else 0.0 for rte in rte_list]
    return rte_vec

def calc_wordnetsim(sub_pred, prem_pred):
    wordnetsim = 0.0
    word_similarity_list = []
    wordFromList1 = wn.synsets(sub_pred)
    wordFromList2 = wn.synsets(prem_pred)
    for w1 in wordFromList1:
        for w2 in wordFromList2:
            if w1.path_similarity(w2) is not None: 
                word_similarity_list.append(w1.path_similarity(w2))
    if(word_similarity_list):
        wordnetsim = max(word_similarity_list)
    return wordnetsim

def calc_ngramsim(sub_pred, prem_pred):
    ngramsim = difflib.SequenceMatcher(None, sub_pred, prem_pred).ratio()
    return ngramsim

def calc_argumentsim(sub_pred, prem_pred, pred_args):
    sub_pred = normalize_token(sub_pred)
    prem_pred = normalize_token(prem_pred)
    if pred_args[sub_pred] == pred_args[prem_pred]:
        return 1.0
    elif pred_args[sub_pred] in pred_args[prem_pred]:
        #ex. sub_goal: play x0, premise: with x0 x1
        return 0.5
    else:
        return 0.0

def check_word2vec(sub_pred, prem_pred):
    threshold = 0.4 #tentatively. to do: optimize threshold
    if unicodedata.category(sub_pred[0]) == "Lo":
        #if Japanese, do URL encode
        sub_pred = urllib.parse.quote(sub_pred)
        prem_pred = urllib.parse.quote(prem_pred)
    process = Popen(\
    'curl http://localhost:5000/word2vec/similarity?w1='+ sub_pred +'\&w2='+ prem_pred, \
    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    similarity, err = process.communicate()
    try:
        if float(similarity.decode()) > threshold:
            return "approx"
    except ValueError:
        return ""


#unused functions
def TryPhraseAbduction_bk(coq_scripts, target):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    direct_proof_scripts, reverse_proof_scripts = [], []
    inference_result_str, all_scripts = "unknown", []
    while inference_result_str == "unknown":
        #continue abduction for phrase acquisition until inference_result_str matches target
        #if target == 'yes':
        #    #entailment proof
        #    inference_result_str, direct_proof_scripts, new_direct_axioms = \
        #        try_phrase_abduction(direct_proof_script,
        #                      previous_axioms=axioms, expected='yes')
        #    current_axioms = axioms.union(new_direct_axioms)
        #elif target == 'no':
        #    #contradiction proof
        #    inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
        #        try_phrase_abduction(reverse_proof_script,
        #                      previous_axioms=axioms, expected='no')
        #    entailment proof
        inference_result_str, direct_proof_scripts, new_direct_axioms = \
            try_phrase_abduction(direct_proof_script,
                              previous_axioms=axioms, expected='yes')
        current_axioms = axioms.union(new_direct_axioms)
        if not inference_result_str == 'yes':
            #contradiction proof
            inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_phrase_abduction(reverse_proof_script,
                              previous_axioms=axioms, expected='no')
            current_axioms = axioms.update(new_reverse_axioms)
        all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms):
            break
        axioms = current_axioms
    return inference_result_str, all_scripts

def make_phrases_from_premises_and_conclusions_ex_(premises, conclusions):
    premises = [p for p in premises if get_pred_from_coq_line(p).startswith('_')]

    p_pred_args = {}
    for p in premises:
        predicate = get_pred_from_coq_line(p, is_conclusion=False)
        args = get_tree_pred_args_ex(p, is_conclusion=False)
        if args is not None:
            p_pred_args[predicate] = args

    c_pred_args = {}
    for c in conclusions:
        predicate = get_pred_from_coq_line(c, is_conclusion=True)
        args = get_tree_pred_args_ex(c, is_conclusion=True)
        if args is not None:
            c_pred_args[predicate] = args

    # Compute relations between arguments as frozensets.
    c_args_preds = defaultdict(set)
    for pred, args in c_pred_args.items():
        for arg in args:
            c_args_preds[frozenset([arg])].add(pred)
        c_args_preds[frozenset(args)].add(pred)
    # from pudb import set_trace; set_trace()
    for args, preds in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
        for targs, _ in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
            # if args.intersection(targs):
            # E.g. if args is {'?4844'} and targs is {'(Acc ?4844)', 'x1'},
            # then we merge the predicates associated to these arguments.
            if any(a in ta for a in args for ta in targs):
                c_args_preds[targs].update(preds)

    exclude_preds_in_conclusion = {
        get_pred_from_coq_line(l, is_conclusion=True) \
            for l in conclusions if not l.startswith('_') and contains_case(l)}

    covered_conclusions = set()
    axioms = set()
    phrase_pairs = []
    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 1:
            premise_preds = [
                p for p, p_args in p_pred_args.items() if set(p_args).issubset(args)]
            premise_preds = sorted([p for p in premise_preds if not contains_case(p)])
            if premise_preds:
                phrase_pairs.append((premise_preds, c_preds)) # Saved phrase pairs for Yanaka-san.
                premise_pred = premise_preds[0]
                for p in c_preds:
                    if p not in covered_conclusions:
                        c_num_args = len(c_pred_args[p])
                        p_num_args = len(p_pred_args[premise_pred])
                        axiom = 'Axiom ax_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                            premise_pred,
                            p,
                            ' '.join('x' + str(i) for i in range(p_num_args)),
                            ' '.join('y' + str(i) for i in range(c_num_args)))
                        axioms.add(axiom)
                        covered_conclusions.add(p)
    #print(phrase_pairs) # this is a list of tuples of lists.
    return axioms

def distinguish_normal_conclusions(conclusions):
    conclusions_normal = []
    for conclusion in conclusions:
        if re.search("\?", conclusion):
            #existential variables contain
            continue
        else:
            #normal variables contain
            conclusions_normal.append(conclusion)
    return conclusions_normal

def get_predicate_case_arguments(premises, conclusion):
    """
    Given the string representations of the premises, where each premises is:
      pX : predicate1 arg1 arg2 arg3
    and the conclusion, which is of the form:
      predicate3 arg2 arg4
    returns a dictionary where the key is a predicate, and the value
    is a list of argument names.
    If the same predicate is found with different arguments, then it is
    labeled as a conflicting predicate and removed from the output.
    Conflicting predicates are typically higher-order predicates, such
    as "Prog".
    """
    pred_args = {}
    pred_trees = []
    for premise in premises:
        try:
            pred_trees.append(
                Tree.fromstring('(' + ' '.join(premise.split()[2:]) + ')'))
        except ValueError:
            continue
    try:
        conclusion_tree = Tree.fromstring('(' + conclusion + ')')
    except ValueError:
        return pred_args
    pred_trees.append(conclusion_tree)
    pred_args_list = []
    for t in pred_trees:
        pred = t.label()
        #if args have case information, extract the pair of case and variables in $args
        #ex. extract Subj x1 as argas in Tree('_lady', [Tree('Subj', ['x1'])])
        #args = t.leaves()
        args = []
        for tt in t:
            args.append(str(tt))
        pred_args_list.append([pred] + args)
    #conflicting_predicates = set()
    count = {}
    for pa in pred_args_list:
        pred = pa[0]
        args = pa[1:]
        if pred not in count:
            count[pred] = 0
            pred_args[pred] = args
        if pred in pred_args and pred_args[pred] != args:
            #conflicting_predicates.add(pred)
            count[pred] += 1
            pred_args[pred+"_"+str(count[pred])] = args        
    #logging.debug('Conflicting predicates: ' + str(conflicting_predicates))
    #for conf_pred in conflicting_predicates:
    #    del pred_args[conf_pred]
    return pred_args

def get_premises_that_partially_match_conclusion_args(premises, conclusion):
    """
    Returns premises where the predicates have at least one argument
    in common with the conclusion.
    In this function, premises containing False are excluded temporarily. For example,
    H0 : forall x : Entity,
       ((_man x /\ (exists e : Event, (_drawing e /\ Subj e = x) /\ True)) /\
        True) /\ (exists e : Event, Subj e = Subj e /\ True) -> False
    to do: how to exclude such an premise perfectly
    """
    candidate_premises = []
    conclusion = re.sub(r'\?([0-9]+)', r'?x\1', conclusion)
    conclusion_args = get_tree_pred_args(conclusion, is_conclusion=True)
    if conclusion_args is None:
        return candidate_premises
    for premise_line in premises:
        # Convert anonymous variables of the form ?345 into ?x345.
        premise_line = re.sub(r'\?([0-9]+)', r'?x\1', premise_line)
        premise_args = get_tree_pred_args(premise_line)
        #print('Conclusion args: ' + str(conclusion_args) +
        #              '\nPremise args: ' + str(premise_args), file=sys.stderr)
        #if tree_contains(premise_args, conclusion_args):
        if premise_args is None or "exists" in premise_line or "=" in premise_line or "forall" in premise_line or "/\\" in premise_line:
            # ignore relation premises temporarily
            continue
        else:
            candidate_premises.append(premise_line)
    #print(candidate_premises, file=sys.stderr)
    return candidate_premises

def make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion_pred, pred_args, expected):
    axioms = set()
    phrase_axioms = get_phrases(premise_preds, conclusion_pred, pred_args, expected)
    axioms.update(set(phrase_axioms))
    return axioms

def get_phrases(premise_preds, conclusion_pred, pred_args, expected):
    #evaluate phrase candidates based on multiple similarities: surface, external knowledge, argument matching
    #in some cases, considering argument matching only is better
    axiom, axioms = "", []
    dist = []
    copyflg = 0
    src_preds = [denormalize_token(p) for p in premise_preds]
    if "False" in conclusion_pred or "=" in conclusion_pred:
        #skip relational subgoals
        return list(set(axioms))
    conclusion_pred = conclusion_pred.split()[0]
    trg_pred = denormalize_token(conclusion_pred)
    for src_pred in src_preds:
        if src_pred == trg_pred:
            #the same premise can be found(copy)
            copyflg = 1
            break
        wordnetsim = calc_wordnetsim(src_pred, trg_pred)
        ngramsim = calc_ngramsim(src_pred, trg_pred)
        argumentsim = calc_argumentsim(src_pred, trg_pred, pred_args)
        #to consider: add categorysim or parse score for smoothing argument match error
        #to consider: how to decide the weight of each info(for now, consider no weight)
        # best score: w_1*wordnetsim + w_2*ngramsim + w_3*argumentsim
        # w_1+w_2+w_3 = 1
        # 0 < wordnetsim < 1, 0 < ngramsim < 1, 0 < argumentsim < 1,
        dist.append(distance.cityblock([1, 1, 1], [wordnetsim, ngramsim, argumentsim]))
    if copyflg == 1:
        axiom = 'Axiom ax_copy_{0} : forall x, _{0} x.'\
        .format(trg_pred)
        axioms.append(axiom)
        return list(set(axioms))

    mindist = dist.index(min(dist))

    best_src_pred = src_preds[mindist]
    best_src_pred_norm = normalize_token(best_src_pred)
    best_src_pred_arg_list = pred_args[best_src_pred_norm]
    best_src_pred_arg = " ".join(best_src_pred_arg_list)

    trg_pred_norm = normalize_token(trg_pred)
    trg_pred_arg_list = pred_args[trg_pred_norm]
    trg_pred_arg = " ".join(trg_pred_arg_list)
        
    total_arg_list = list(set(best_src_pred_arg_list + trg_pred_arg_list))
    total_arg_list = check_case_from_list(total_arg_list)
    total_arg = " ".join(total_arg_list)
        
    axiom = 'Axiom ax_phrase{0}{1} : forall {2}, {0} {3} -> {1} {4}.'\
            .format(best_src_pred_norm, trg_pred_norm, total_arg, best_src_pred_arg, trg_pred_arg)
    print("premise_pred:{0}, conclusion_pred:{1}, pred_args:{2}, axiom:{3}".format(premise_preds, conclusion_pred, pred_args, axiom), file=sys.stderr)

    # to do: consider how to inject antonym axioms
    axioms.append(axiom)
    return list(set(axioms))

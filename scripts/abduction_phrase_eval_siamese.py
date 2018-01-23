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

from abduction_tools import *
from knowledge import get_tokens_from_xml_node, get_lexical_relations_from_preds
from normalization import denormalize_token, normalize_token
from nltk.corpus import wordnet as wn
import difflib
from scipy.spatial import distance
from semantic_tools import is_theorem_defined
from tactics import get_tactics
from tree_tools import is_string
import pandas as pd
from linguistic_tools import linguistic_relationship, get_wordnet_cascade
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

model = load_model('./phrase_results2/siamese_mlp_model.mm')

class AxiomsPhraseEval(object):
    """
    Evaluate RTE using phrasal axioms extracted from Training dataset
    """
    def __init__(self):
        pass

    def attempt(self, coq_scripts, doc, context=None):
        return TryPhraseAbduction(coq_scripts, doc)

def TryPhraseAbduction(coq_scripts, doc):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    direct_proof_scripts, reverse_proof_scripts = [], []
    inference_result_str, all_scripts = "unknown", []
    while True:
        #entailment proof
        inference_result_str, direct_proof_scripts, new_direct_axioms = \
            try_phrase_abduction(direct_proof_script, doc,
                        previous_axioms=axioms, expected='yes')
        current_axioms = axioms.union(new_direct_axioms)
        if inference_result_str == 'unknown':
            #contradiction proof
            inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_phrase_abduction(reverse_proof_script, doc,
                              previous_axioms=axioms, expected='no')
            current_axioms = axioms.union(new_reverse_axioms)
        all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms) or inference_result_str != 'unknown':
            break
        axioms = current_axioms
    return inference_result_str, all_scripts
    
def try_phrase_abduction(coq_script, doc, previous_axioms=set(), expected='yes'):
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
    if is_theorem_almost_defined(output_lines):
        return expected, [new_coq_script], previous_axioms
    premise_lines = get_premise_lines(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    conclusions = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusions:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_phrase_axioms(premise_lines, conclusions, doc, output_lines, expected, coq_script_debug)
    #axioms = filter_wrong_axioms(axioms, coq_script) temporarily
    axioms = axioms.union(previous_axioms)
    new_coq_script = insert_axioms_in_coq_script(axioms, coq_script_debug)
    #print(coq_script_debug, new_coq_script)
    process = Popen(
        new_coq_script,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    inference_result_str = expected if is_theorem_almost_defined(output_lines) else 'unknown'
    return inference_result_str, [new_coq_script], axioms

def make_phrase_axioms(premises, conclusions, doc, coq_output_lines=None, expected='yes', coq_script_debug=None):
    #check premises and sub-goals, search for their relations from sqlite, select axioms
    axioms = set()
    axioms = make_phrases_from_premises_and_conclusions_ex(premises, conclusions, doc, coq_script_debug, expected)

    #for conclusion in conclusions:
    #    matching_premises, conclusion = get_premises_that_partially_match_conclusion_args(premises, conclusion)
    #    premise_preds = [premise.split()[2] for premise in matching_premises]
    #    pred_args = get_predicate_case_arguments(matching_premises, conclusion)
    #    axioms.update(make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion, pred_args, expected))
    #    if not axioms:
    #        failure_log = make_failure_log(
    #            conclusion, premise_preds, conclusion, premises, coq_output_lines)
    #        print(json.dumps(failure_log), file=sys.stderr)
    return axioms

def make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion_pred, pred_args, expected):
    axioms = set()
    phrase_axioms = get_phrases(premise_preds, conclusion_pred, pred_args, expected)
    axioms.update(set(phrase_axioms))
    return axioms

def get_conclusion_lines(coq_output_lines):
    #print(coq_output_lines, file=sys.stderr)
    conclusion_lines = []
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(coq_output_lines)
    if not line_index_last_conclusion_sep:
        return None
    for line in coq_output_lines[line_index_last_conclusion_sep+1:]:
        if re.search('Toplevel', line):
            return conclusion_lines
        elif line == '':
            continue
        elif '=' in line:
            #skip relational sub-goals
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

def get_phrases(premise_preds, conclusion_pred, pred_args, expected):
    #evaluate phrase candidates based on multiple similarities: surface, external knowledge, argument matching
    #in some cases, considering argument matching only is better
    axiom, axioms = "", []
    copyflg = 0
    src_preds = [denormalize_token(p) for p in premise_preds]
    if "False" in conclusion_pred or "=" in conclusion_pred:
        #skip relational subgoals
        return list(set(axioms))
    conclusion_pred = conclusion_pred.split()[0]
    trg_pred = denormalize_token(conclusion_pred)
    for src_pred in src_preds:
        #if src_pred == trg_pred:
            #the same premise can be found(copy)
            #copyflg = 1
            #break
        nor_src_pred = normalize_token(src_pred)
        allarg_list = list(set(pred_args[conclusion_pred] + pred_args[nor_src_pred]))
        allarg_list = check_case_from_list(allarg_list)
        allarg = " ".join(allarg_list)
        axiom = search_axioms_from_db(src_pred, trg_pred, pred_args[conclusion_pred], pred_args[nor_src_pred], allarg)
        #to do: select best axiom from premises(or make phrasal axiom coq_script)
        if axiom:
            axioms.append(axiom)
    #if copyflg == 1:
    #    axiom = 'Axiom ax_copy_{0} : forall x, _{0} x.'\
    #    .format(trg_pred)
    #    axioms.append(axiom)
    print("premise_pred:{0}, conclusion_pred:{1}, pred_args:{2}, axiom:{3}".format(premise_preds, conclusion_pred, pred_args, axioms), file=sys.stderr)
    # to do: consider how to inject antonym axioms

    return list(set(axioms))

def search_axioms_from_db(src_pred, trg_pred, sub_arg_list, prem_arg_list, allarg):
    import sqlite3
    coq_axiom = ""
    sub_arg = " ".join(sub_arg_list)
    prem_arg = " ".join(prem_arg_list)
    con = sqlite3.connect('./sick_phrase.sqlite3')
    cur = con.cursor()
    #1. search for premise-subgoal relations from sqlite
    df = pd.io.sql.read_sql_query('select * from {table} where premise = \"{src_pred}\" and subgoal = \"{trg_pred}\"'\
        .format(table='axioms', trg_pred=trg_pred, src_pred=src_pred), con)
    
    #select axioms from argument information if there are multiple candidate axioms in database.
    #But there are many duplicates in database for now. 
    #Temporarily, return the first record as the best candidate axiom
    if not df.empty:
        db_premise = df.loc[0, ["premise"]].values[0]
        db_subgoal = df.loc[0, ["subgoal"]].values[0]
        db_prem_arg = df.loc[0, ["prem_arg"]].values[0]
        db_sub_arg = df.loc[0, ["sub_arg"]].values[0]
        db_kind = df.loc[0, ["kind"]].values[0]
        db_allarg = df.loc[0, ["allarg"]].values[0]
        coq_axiom = "Axiom ax_{0}_{1}_{2} : forall {3}, _{1} {4} -> _{2} {5}.".format(
                        db_kind,
                        db_premise,
                        db_subgoal,
                        allarg,
                        prem_arg,
                        sub_arg)
    #else:
        #2. if no record is found, search for subgoal-premise relations from sqlite again
        #df2 = pd.io.sql.read_sql_query('select * from {table} where premise = \"{trg_pred}\" and subgoal = \"{src_pred}\"'\
        #     .format(table='axioms', trg_pred=trg_pred, src_pred=src_pred), con)
        #if not df2.empty:
        #    db_subgoal = df2.loc[0, ["premise"]].values[0]
        #    db_premise = df2.loc[0, ["subgoal"]].values[0]
        #    db_sub_arg = df2.loc[0, ["prem_arg"]].values[0]
        #    db_prem_arg = df2.loc[0, ["sub_arg"]].values[0]
        #    db_kind = df2.loc[0, ["kind"]].values[0]
        #    db_allarg = df2.loc[0, ["allarg"]].values[0]
        #    coq_axiom = "Axiom ax_{0}_{1}_{2} : forall {3}, _{1} {4} -> _{2} {5}.".format(
        #                    db_kind,
        #                    db_premise,
        #                    db_subgoal,
        #                    allarg,
        #                    prem_arg,
        #                    sub_arg)
    print("coq_axioms_from_db: {0}, trg_pred: {1}, src_pred: {2}".format(coq_axiom, trg_pred, src_pred), file=sys.stderr)
    con.close()
    return coq_axiom


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
    conclusions = get_conclusion_lines(output_lines)
    print("conclusion:{0}".format(conclusions), file=sys.stderr)
    subgoalflg = 0
    if conclusions is None:
        return False
    if len(conclusions) > 0:
        for conclusion in conclusions:
            if not "=" in conclusion:
                subgoalflg = 1
            if "No more subgoals" in conclusion:
                return True
    if subgoalflg == 1:
        return False
    else:
        return True

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
    conclusion = re.sub(r'\?([0-9]+)', r'x\1', conclusion)
    conclusion_args = get_tree_pred_args(conclusion, is_conclusion=True)
    if conclusion_args is None:
        return candidate_premises, conclusion
    for premise_line in premises:
        # Convert anonymous variables of the form ?345 into ?x345.
        premise_line = re.sub(r'\?([0-9]+)', r'x\1', premise_line)
        premise_args = get_tree_pred_args(premise_line)
        #print('Conclusion args: ' + str(conclusion_args) +
        #              '\nPremise args: ' + str(premise_args), file=sys.stderr)
        #if tree_contains(premise_args, conclusion_args):
        if premise_args is None or "exists" in premise_line or "=" in premise_line or "forall" in premise_line or "/\\" in premise_line:
            # ignore relation premises temporarily
            continue
        else:
            candidate_premises.append(premise_line)
    return candidate_premises, conclusion

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
    if not is_conclusion:
        line = ' '.join(line.split()[2:])
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
        if len(line.split()) > 2:
            return line.split()[2]
        else:
            return line.split()[0]
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
                #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                c_ph_word = re.sub("_", "", case_c_pred)
                p_ph_word = re.sub("_", "", p_pred)
                typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                wordnetrel, antonym = check_wordnetrel(c_ph_word, p_ph_word)
                #rte = check_rte(expected)
                feature = simlist + wordnetrel

                axiom = 'Axiom ax_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                    p_pred,
                    case_c_pred,
                    "x0",
                    "y0")
                if antonym == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                    #check if entailment axiom was generated
                    exist_axioms = [axiom for axiom in axioms if p in axiom]
                    if len(exist_axioms) > 0:
                        for exist_axiom in exist_axioms:
                            axioms.remove(exist_axiom)
                    #antonym axiom for event predicates
                    axiom = 'Axiom ax_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        p_pred,
                        case_c_pred)
                elif antonym == "antonym":
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
                features[antonym+p_pred+case_c_pred] = feature


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

                            #extract features of these candidates by type, ngram, WN(sim, relation), W2V, RTE_gold
                            c_ph_word = re.sub("_", "", p)
                            p_ph_word = re.sub("_", "", premise_pred)
                            typesim = calc_typesim(check_types(c_ph_word, type_lists), check_types(p_ph_word, type_lists))
                            ngramsim = calc_ngramsim(c_ph_word, p_ph_word)
                            wordnetsim = calc_wordnetsim(c_ph_word, p_ph_word)
                            word2vecsim = calc_word2vecsim(c_ph_word, p_ph_word)
                            simlist = [typesim, ngramsim, wordnetsim, word2vecsim]
                            wordnetrel, antonym = check_wordnetrel(c_ph_word, p_ph_word)
                            #rte = check_rte(expected)
                            feature = simlist + wordnetrel

                            axiom = 'Axiom ax_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                                premise_pred,
                                p,
                                ' '.join('x' + str(i) for i in range(p_num_args)),
                                ' '.join('y' + str(i) for i in range(c_num_args)))
                            if antonym == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
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
                            elif antonym == "antonym":
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
                        features[antonym+premise_pred+p] = feature
                        
    #select premise features whose similarity are max and min
    sum_dist = {}
    feature_dist = {}
    return_feature = {}
    new_axioms = []
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
        predict = np.round(model.predict([t_vector, h_vector, np.array([max_feature + min_feature]).reshape(1, 28)]))
        #if axiom classifier is 1, create axioms about the subgoal
        if int(predict) == 1:
            pattern = re.compile(s+" : forall")
            for a in axioms:
                if re.search(pattern, a):
                    new_axioms.append(a)

    #print(phrase_pairs) # this is a list of tuples of lists.

    return set(new_axioms)

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
    if relation == "antonym":
        antonym = "antonym"
    rel_vec = [1.0 if rel == relation else 0.0 for rel in rel_list]
    return rel_vec, antonym

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
        #if args have case information, extract the pair of case and variables in args
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
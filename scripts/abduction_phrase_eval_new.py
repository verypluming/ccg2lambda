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

class AxiomsPhraseEval(object):
    """
    Evaluate RTE using phrasal axioms extracted from Training dataset
    """
    def __init__(self):
        pass

    def attempt(self, coq_scripts, doc, context=None):
        return TryPhraseAbduction(coq_scripts)

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

def TryPhraseAbduction(coq_scripts):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    direct_proof_scripts, reverse_proof_scripts = [], []
    inference_result_str, all_scripts = "unknown", []
    while True:
        #entailment proof
        inference_result_str, direct_proof_scripts, new_direct_axioms = \
            try_phrase_abduction(direct_proof_script,
                        previous_axioms=axioms, expected='yes')
        current_axioms = axioms.union(new_direct_axioms)
        if inference_result_str == 'unknown':
            #contradiction proof
            inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_phrase_abduction(reverse_proof_script,
                              previous_axioms=axioms, expected='no')
            current_axioms = axioms.union(new_reverse_axioms)
        all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms) or inference_result_str != 'unknown':
            break
        axioms = current_axioms
    return inference_result_str, all_scripts
    
def try_phrase_abduction(coq_script, previous_axioms=set(), expected='yes'):
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
    premise_lines = get_premise_lines_ex(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    conclusions = get_conclusion_lines_ex(output_lines)
    if not premise_lines or not conclusions:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_phrase_axioms(premise_lines, conclusions, output_lines, expected)
    #axioms = filter_wrong_axioms(axioms, coq_script) temporarily
    axioms = axioms.union(previous_axioms)
    new_coq_script = insert_axioms_in_coq_script(axioms, coq_script_debug)
    process = Popen(
        new_coq_script,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    inference_result_str = expected if is_theorem_almost_defined(output_lines) else 'unknown'
    return inference_result_str, [new_coq_script], axioms

def make_phrase_axioms(premises, conclusions, coq_output_lines=None, expected='yes'):
    #check premises and sub-goals, search for their relations from sqlite, select axioms
    axioms = set()
    
    for conclusion in conclusions:
        axioms.update(make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion, pred_args, expected))
        if not axioms:
            failure_log = make_failure_log(
                conclusion, premise_preds, conclusion, premises, coq_output_lines)
            print(json.dumps(failure_log), file=sys.stderr)
    return axioms

def make_phrase_axioms_from_premises_and_conclusions(premise_preds, conclusion_pred, pred_args, expected):
    axioms = set()
    phrase_axioms = get_phrases(premise_preds, conclusion_pred, pred_args, expected)
    axioms.update(set(phrase_axioms))
    return axioms

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
        #elif '=' in line:
        #    #skip relational sub-goals
        #    continue
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
    axiom, axioms = "", []
    kinds, dist = [], []
    copyflg = 0
    src_preds = [denormalize_token(p) for p in premise_preds]
    if "False" in conclusion_pred or "=" in conclusion_pred:
        #skip relational subgoals
        return list(set(axioms))
    conclusion_pred = conclusion_pred.split()[0]
    trg_pred = denormalize_token(conclusion_pred)
    axioms = search_axioms_from_db(src_preds, trg_pred, pred_args[conclusion_pred], pred_args)

    print("premise_pred:{0}, conclusion_pred:{1}, pred_args:{2}, axiom:{3}".format(premise_preds, conclusion_pred, pred_args, axioms), file=sys.stderr)
    return list(set(axioms))

def search_axioms_from_db(src_preds, trg_pred, sub_arg_list, pred_args):
    import sqlite3
    axiom = ""
    candidates = defaultdict(list)
    con = sqlite3.connect('./sick_phrase.sqlite3')
    cur = con.cursor()
    for src_pred in src_preds:
        #1. search for premise-subgoal relations from sqlite
        df = pd.io.sql.read_sql_query('select * from {table} where premise like \"{src_preds}\" and subgoal = \"{trg_pred}\"'\
            .format(table='axioms', trg_pred=trg_pred, src_pred=src_preds), con)
        #select axioms from argument information if there are multiple candidate axioms in database.
        if not df.empty:
            db_premise = df.loc[0, ["premise"]].values[0]
            db_subgoal = df.loc[0, ["subgoal"]].values[0]
            db_prem_arg = df.loc[0, ["prem_arg"]].values[0]
            db_sub_arg = df.loc[0, ["sub_arg"]].values[0]
            db_kind = df.loc[0, ["kind"]].values[0]
            db_allarg = df.loc[0, ["allarg"]].values[0]
            candidates[db_subgoal].append([db_premise, db_subgoal, db_prem_arg, db_sub_arg, db_kind, db_allarg])

    if candidates:
        for i, candidate in enumarate(candidates):
            if i == 0:
                axiom = 'Axiom ax_phrase_norelation_{0}_{1} : forall {2}, '.format(
                db_premise,
                db_subgoal,
                db_allarg,
                )

            axiom += candidate[0]+" "+candidate[2]+" -> "
            if candidate[4] == "antonym" and "Event -> Prop" in check_types(candidate[1], type_lists):
                #antonym axiom for event predicates
                axiom = 'Axiom ax_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        premise_pred,
                        p)
                break
            elif candidate[4] == "antonym" and "Entity -> Prop" in check_types(candidate[1], type_lists):
                #antonym axiom for entity->prop predicates
                axiom = 'Axiom ax_phrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F x -> F y  -> False.'.format(
                        premise_pred,
                        p)
                        break
            elif candidate[4] == "antonym":
                #antonym axiom for entity predicates
                axiom = 'Axiom ax_phrase_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                        premise_pred,
                        p)
                        break
        if candidate[4] != "antonym":
            axiom += db_subgoal+" "+candidate[3]+"."


    print("coq_axioms_from_db: {0}, conc_pred: {1}, prem_pred: {2}, conc_arg: {3}, prem_arg: {4}".format(axiom, trg_pred, src_pred, sub_arg, prem_arg), file=sys.stderr)
    con.close()
    return axiom


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
    conclusions = get_conclusion_lines_ex(output_lines)
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


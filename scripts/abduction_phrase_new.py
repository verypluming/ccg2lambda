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
from itertools import chain
import urllib.parse
import unicodedata

class AxiomsPhrase(object):
    """
    Create phrasal axioms 
    """
    def __init__(self):
        pass

    def attempt(self, coq_scripts, doc, target, context=None):
        return TryPhraseAbduction(coq_scripts, target)

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


def TryPhraseAbduction(coq_scripts, target):
    #assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms, current_axioms = set(), set()
    direct_proof_scripts, reverse_proof_scripts = [], []
    inference_result_str, all_scripts = "unknown", []
    while inference_result_str == "unknown":
        #continue abduction for phrase acquisition until inference_result_str matches target
        if target == 'yes':
            #entailment proof
            inference_result_str, direct_proof_scripts, new_direct_axioms = \
                try_phrase_abduction(direct_proof_script,
                            previous_axioms=axioms, expected='yes')
            current_axioms = axioms.union(new_direct_axioms)
        elif target == 'no':
            #entailment proof and then contradiction proof
            inference_result_str, direct_proof_scripts, new_direct_axioms = \
                try_phrase_abduction(direct_proof_script,
                            previous_axioms=axioms, expected='yes')
            current_axioms = axioms.union(new_direct_axioms)
            if inference_result_str == 'unknown':
                #contradiction proof
                inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                    try_phrase_abduction(reverse_proof_script,
                                previous_axioms=axioms, expected='no')
            current_axioms.union(new_reverse_axioms)
        all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms):
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
    premise_lines = get_premise_lines_ex(output_lines)
    conclusion = get_conclusion_lines_ex(output_lines)
    if is_theorem_almost_defined(output_lines):
        #positive label
        features_log = {"proof": expected, "premise": premise_lines, "subgoals": conclusion}
        print(json.dumps(features_log), file=sys.stderr)
        return expected, [new_coq_script], previous_axioms
    #premise_lines = get_premise_lines(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    #conclusion = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusion:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_phrase_axioms(premise_lines, conclusion, output_lines, expected, coq_script_debug)
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
    features_log = {"proof": expected, "premise": premise_lines, "subgoals": conclusion}
    print(json.dumps(features_log), file=sys.stderr)
    return inference_result_str, [new_coq_script], axioms

def make_phrase_axioms(premises, conclusions, coq_output_lines=None, expected='yes', coq_script_debug=None):
    axioms = set()
    axioms = make_phrases_from_premises_and_conclusions_ex(premises, conclusions, coq_script_debug, expected)

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

def make_case_phrases(c_pred_args, p_pred_args, c_pred_args_id, p_pred_args_id, type_lists, case_c_pred, axioms, covered_conclusions):
    case_c_arg = c_pred_args[case_c_pred][0][0]
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
        relation = ""
        axiom = 'Axiom ax_casephrase_norelation{0}{1} : forall {2}, '.format(
                "".join(case_p_preds),
                case_c_pred,
                ' '.join('x' + str(i) for i in list(set(case_p_preds_args_id+c_pred_args_id[case_c_pred]))),
                )
        for case_p_pred in case_p_preds:
            #create axioms with premises which have the same case 
            axiom += case_p_pred+" "+" ".join('x' + str(i) for i in p_pred_args_id[case_p_pred])+" -> "
            c_ph_word = re.sub("_", "", case_c_pred)
            p_ph_word = re.sub("_", "", case_p_pred)
            wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)

            if relation == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
                #check if entailment axiom was generated
                exist_axioms = [axiom for axiom in axioms if case_c_pred in axiom]
                if len(exist_axioms) > 0:
                    for exist_axiom in exist_axioms:
                        axioms.remove(exist_axiom)
                #antonym axiom for event predicates
                axiom = 'Axiom ax_casephrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                break
            elif relation == "antonym" and "Entity -> Prop" in check_types(c_ph_word, type_lists):
                #check if entailment axiom was generated
                exist_axioms = [axiom for axiom in axioms if case_c_pred in axiom]
                if len(exist_axioms) > 0:
                    for exist_axiom in exist_axioms:
                        axioms.remove(exist_axiom)
                #antonym axiom for entity->prop predicates
                axiom = 'Axiom ax_casephrase_antonym{0}{1} : forall F x y, {0} x -> {1} y -> F x -> F y  -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                break
            elif relation == "antonym":
                #check if entailment axiom was generated
                exist_axioms = [axiom for axiom in axioms if case_c_pred in axiom]
                if len(exist_axioms) > 0:
                    for exist_axiom in exist_axioms:
                        axioms.remove(exist_axiom)
                #antonym axiom for entity predicates
                axiom = 'Axiom ax_casephrase_antonym{0}{1} : forall x, {0} x -> {1} x -> False.'.format(
                        case_p_pred,
                        case_c_pred)
                break
        if relation != "antonym":
            axiom += case_c_pred+" "+' '.join('x' + str(i) for i in c_pred_args_id[case_c_pred])+"."
        covered_conclusions.add(case_c_pred)
        axioms.append(axiom)
    return axioms, covered_conclusions
        
def make_phrases(c_preds, premise_preds, c_pred_args_id, p_pred_args_id, type_lists, axioms, covered_conclusions):
    premise_preds_args_id = []
    for p in c_preds:
        relations = {}
        for premise_pred in premise_preds:
            premise_preds_args_id.extend(p_pred_args_id[premise_pred])
            c_ph_word = re.sub("_", "", p)
            p_ph_word = re.sub("_", "", premise_pred)
            wordnetrel, relation = check_wordnetrel(c_ph_word, p_ph_word)
            relations[premise_pred] = relation
        if p not in covered_conclusions:
            axiom = 'Axiom ax_phrase_norelation{0}{1} : forall {2}, '.format(
                "".join(premise_preds),
                p,
                ' '.join('x' + str(i) for i in list(set(premise_preds_args_id+c_pred_args_id[p])))
                )
            for premise_pred in premise_preds:
                axiom += premise_pred+" "+" ".join('x' + str(i) for i in p_pred_args_id[premise_pred])+" -> "
                c_ph_word = re.sub("_", "", p)
                p_ph_word = re.sub("_", "", premise_pred)
                if relations[premise_pred] == "antonym" and "Event -> Prop" in check_types(c_ph_word, type_lists):
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
                elif relations[premise_pred] == "antonym" and "Entity -> Prop" in check_types(c_ph_word, type_lists):
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
                elif relations[premise_pred] == "antonym":
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
            if not "antonym" in relations.values():
                axiom += p+" "+' '.join('x' + str(i) for i in c_pred_args_id[p])+"."
            covered_conclusions.add(p)
            axioms.append(axiom)
    return axioms, covered_conclusions

def make_phrases_from_premises_and_conclusions_ex(premises, conclusions, coq_script_debug=None, expected="yes"):
    coq_lists, param_lists, type_lists = [], [], []
    if coq_script_debug:
        coq_lists = coq_script_debug.split("\n")
        param_lists = [re.sub("Parameter ", "", coq_list) for coq_list in coq_lists if re.search("Parameter", coq_list)]
        type_lists = [param_list.split(":") for param_list in param_lists]
    covered_conclusions = set()
    axioms = []
    phrase_pairs = []
    premises = [p for p in premises if get_pred_from_coq_line(p).startswith('_') and check_decomposed(p)]
    args_id = defaultdict(lambda: len(args_id))
    p_pred_args_id = {}
    c_pred_args_id = {}
    p_pred_args = {}

    for p in premises:
        predicate = get_pred_from_coq_line(p, is_conclusion=False)
        args = get_tree_pred_args_ex(p, is_conclusion=False)
        if args is not None:
            p_pred_args[predicate] = args
            #p_pred_args[predicate].append(args) # List of lists of args.
            p_pred_args_id[predicate] = [args_id[arg] for arg in args]

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

    exclude_preds_in_conclusion = {
        get_pred_from_coq_line(l, is_conclusion=True) \
        for l in conclusions if contains_case(l) and not l.startswith('_')}

    #create axioms about sub-goals containing case information
    case_c_preds = [c for c, c_args in c_pred_args.items() if contains_case(str(c_args)) and len(c_args[0]) == 1]
    for case_c_pred in case_c_preds:
        axioms, covered_conclusions = make_case_phrases(c_pred_args, p_pred_args, c_pred_args_id, p_pred_args_id, type_lists, case_c_pred, axioms, covered_conclusions)

    #create axioms about sub-goals without case information
    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 0:
            premise_preds = sorted([
                 prem for prem, p_args in p_pred_args.items() if set(p_args).issuperset(args) and not contains_case(str(p_args))])
            #print("premise_preds: {0}, c_preds: {1}, p_pred_args: {2}, args: {3}, c_args_preds: {4}".format(premise_preds, c_preds, p_pred_args, args, c_args_preds))
            if premise_preds:
                axioms, covered_conclusions = make_phrases(c_preds, premise_preds, c_pred_args_id, p_pred_args_id, type_lists, axioms, covered_conclusions)
            else:
                #if perfect match is not exist, make extended phrase set
                premise_preds = sorted([
                    prem for prem, p_args in p_pred_args.items() if set(p_args).issubset(args) and not contains_case(str(p_args))])
                if premise_preds:
                    axioms, covered_conclusions = make_phrases(c_preds, premise_preds, c_pred_args_id, p_pred_args_id, type_lists, axioms, covered_conclusions)

    # from pudb import set_trace; set_trace()
    for args, preds in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
        for targs, _ in sorted(c_args_preds.items(), key=lambda x: len(x[0])):
            # if args.intersection(targs):
            # E.g. if args is {'?4844'} and targs is {'(Acc ?4844)', 'x1'},
            # then we merge the predicates associated to these arguments.
            if any(a in ta for a in args for ta in targs):
                c_args_preds[targs].update(preds)

    #create axioms about sub-goals without case information again
    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 0:
            premise_preds = sorted([
                 prem for prem, p_args in p_pred_args.items() if set(p_args).issuperset(args) and not contains_case(str(p_args))])
            #print("premise_preds: {0}, c_preds: {1}, p_pred_args: {2}, args: {3}, c_args_preds: {4}".format(premise_preds, c_preds, p_pred_args, args, c_args_preds))
            if premise_preds:
                axioms, covered_conclusions = make_phrases(c_preds, premise_preds, c_pred_args_id, p_pred_args_id, type_lists, axioms, covered_conclusions)
            else:
                #if perfect match is not exist, make extended phrase set
                premise_preds = sorted([
                    prem for prem, p_args in p_pred_args.items() if set(p_args).issubset(args) and not contains_case(str(p_args))])
                if premise_preds:
                    axioms, covered_conclusions = make_phrases(c_preds, premise_preds, c_pred_args_id, p_pred_args_id, type_lists, axioms, covered_conclusions)


    return set(axioms)


def check_types(pred, type_lists):
    for type_list in type_lists:
        if pred in type_list[0]:
            return type_list[1]

def check_wordnetrel(sub_pred, prem_pred):
    antonym = "phrase"
    rel_list = ['copy', 'inflection', 'derivation', 'synonym', 'antonym', 'hypernym', 'hyponym', 'sister', 'cousin', 'similar']
    relations = linguistic_relationship(prem_pred, sub_pred)
    relation = get_wordnet_cascade(relations)
    rel_vec = [1.0 if rel == relation else 0.0 for rel in rel_list]
    if relation is None:
        relation = "norelation"
    return rel_vec, relation


# -*- coding: utf-8 -*-
#
#  Copyright 2017 Pascual Martinez-Gomez and Hitomi Yanaka
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

from collections import OrderedDict
import json
import logging
import re
from subprocess import Popen
import subprocess
import sys

from nltk import Tree

from knowledge import get_lexical_relations_from_preds, get_approx_relations_from_preds
from normalization import denormalize_token, normalize_token
from tactics import get_tactics
from tree_tools import tree_or_string, tree_contains
import unicodedata
import pandas as pd

def is_theorem_defined(output_lines):
    """
    Check whether the string "is defined" appears in the output of coq.
    In that case, we return True. Otherwise, we return False.
    """
    for output_line in output_lines:
        if len(output_line) > 2 and 'is defined' in (' '.join(output_line[-2:])):
            return True
    return False


def is_theorem_error(output_lines):
    """
    Errors in the construction of a theorem (type mismatches in axioms, etc.)
    are signaled using the symbols ^^^^ indicating where the error is.
    We simply search for that string.
    """
    return any('^^^^' in o for ol in output_lines for o in ol)

def is_theorem_error2(output_lines):
    if any('Error' in o for ol in output_lines for o in ol):
        if any('environment' in o for ol in output_lines for o in ol):
            return True
        else:
            return False


def find_final_subgoal_line_index(coq_output_lines):
    indices = [i for i, line in enumerate(coq_output_lines)
               if line.endswith('subgoal')]
    if not indices:
        return None
    return indices[-1]


def find_final_conclusion_sep_line_index(coq_output_lines):
    indices = [i for i, line in enumerate(coq_output_lines)
               if line.startswith('===') and line.endswith('===')]
    if not indices:
        return None
    return indices[-1]


def get_premise_lines(coq_output_lines):
    premise_lines = []
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(
        coq_output_lines)
    if not line_index_last_conclusion_sep:
        return premise_lines
    for line in coq_output_lines[line_index_last_conclusion_sep - 1:0:-1]:
        if line == "":
            return premise_lines
        else:
            premise_lines.append(line)
    return premise_lines


def get_conclusion_line(coq_output_lines):
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(
        coq_output_lines)
    if not line_index_last_conclusion_sep:
        return None
    return coq_output_lines[line_index_last_conclusion_sep + 1]

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
        elif re.search('subgoal', line):
            continue
        elif re.search('repeat nltac_base', line):
            return conclusion_lines
        else:
            conclusion_lines.append(line)
    return conclusion_lines

def get_premises_that_match_conclusion_args_(premises, conclusion):
    """
    Returns premises where the predicates have at least one argument
    in common with the conclusion.
    This function was used for EACL 2017.
    """
    conclusion_terms = [c.strip(')(') for c in conclusion.split()]
    conclusion_args = set(conclusion_terms[1:])
    candidate_premises = []
    for premise in premises:
        premise_terms = [p.strip(')(') for p in premise.split()[2:]]
        premise_args = set(premise_terms[1:])
        logging.debug('Conclusion args: ' + str(conclusion_args) +
                      '\nPremise args: ' + str(premise_args))
        if premise_args.intersection(conclusion_args):
            candidate_premises.append(premise)
    return candidate_premises


def get_premises_that_match_conclusion_args(premises, conclusion):
    """
    Returns premises where the predicates have at least one argument
    in common with the conclusion.
    """
    candidate_premises = []
    conclusion = re.sub(r'\?([0-9]+)', r'?x\1', str(conclusion))
    conclusion_args = get_tree_pred_args(conclusion, is_conclusion=True)
    if conclusion_args is None:
        return candidate_premises
    for premise_line in premises:
        # Convert anonymous variables of the form ?345 into ?x345.
        premise_line = re.sub(r'\?([0-9]+)', r'?x\1', premise_line)
        premise_args = get_tree_pred_args(premise_line)
        logging.debug('Conclusion args: ' + str(conclusion_args) +
                      '\nPremise args: ' + str(premise_args))
        if tree_contains(premise_args, conclusion_args):
            candidate_premises.append(premise_line)
    return candidate_premises


def make_failure_log(conclusion_pred, premise_preds, conclusion, premises,
                     coq_output_lines=None):
    """
    Produces a dictionary with the following structure:
    {"unproved sub-goal" : "sub-goal_predicate",
     "matching premises" : ["premise1", "premise2", ...],
     "raw sub-goal" : "conclusion",
     "raw premises" : ["raw premise1", "raw premise2", ...]}
    Raw sub-goal and raw premises are the coq lines with the premise
    internal name and its predicates. E.g.
    H : premise (Acc x1)
    Note that this function is not capable of returning all unproved
    sub-goals in coq's stack. We only return the top unproved sub-goal.
    """
    failure_log = OrderedDict()
    conclusion_base = denormalize_token(conclusion_pred)
    failure_log["unproved sub-goal"] = conclusion_base
    premises_base = [denormalize_token(p) for p in premise_preds]
    failure_log["matching premises"] = premises_base
    failure_log["raw sub-goal"] = conclusion
    failure_log["raw premises"] = premises
    premise_preds = []
    for p in premises:
        try:
            pred = p.split()[2]
        except:
            continue
        if pred.startswith('_'):
            premise_preds.append(denormalize_token(pred))
    failure_log["all premises"] = premise_preds
    failure_log["other sub-goals"] = get_subgoals_from_coq_output(
        coq_output_lines, premises)
    failure_log["type error"] = has_type_error(coq_output_lines)
    failure_log["open formula"] = has_open_formula(coq_output_lines)
    return failure_log


def has_type_error(coq_output_lines):
    for line in coq_output_lines:
        if 'has type' in line and 'while it is expected to have type' in line:
            return 'yes'
    return 'no'


def has_open_formula(coq_output_lines):
    for line in coq_output_lines:
        if 'The type of this term is a product while it is expected to be' in line:
            return 'yes'
        if '(fun F' in line:
            return 'yes'
    return 'no'


def get_subgoals_from_coq_output(coq_output_lines, premises):
    """
    When the proving is halted due to unprovable sub-goals,
    Coq produces an output similar to this:

    2 subgoals

      H1 : True
      H4 : True
      x1 : Event
      H6 : True
      H3 : _play x1
      H : _two (Subj x1)
      H2 : _man (Subj x1)
      H0 : _table (Acc x1)
      H5 : _tennis (Acc x1)
      ============================
       _ping (Acc x1)

    subgoal 2 is:
      _pong (Acc x1)

    This function returns the remaining sub-goals ("_pong" in this example).
    """
    subgoals = []
    subgoal_index = -1
    for line in coq_output_lines:
        if line.strip() == '':
            continue
        line_tokens = line.split()
        if subgoal_index > 0:
            subgoal_line = line
            subgoal_tokens = subgoal_line.split()
            subgoal_pred = subgoal_tokens[0]
            if subgoal_index in [s['index'] for s in subgoals]:
                # This sub-goal has already appeared and is recorded.
                subgoal_index = -1
                continue
            subgoal = {
                'predicate': denormalize_token(line_tokens[0]),
                'index': subgoal_index,
                'raw': subgoal_line}
            matching_premises = get_premises_that_match_conclusion_args(
                premises, subgoal_line)
            subgoal['matching raw premises'] = matching_premises
            premise_preds = [
                denormalize_token(premise.split()[2]) for premise in matching_premises]
            subgoal['matching premises'] = premise_preds
            subgoals.append(subgoal)
            subgoal_index = -1
        if len(line_tokens) >= 3 and line_tokens[0] == 'subgoal' and line_tokens[2] == 'is:':
            subgoal_index = int(line_tokens[1])
    return subgoals


def make_axioms_from_premises_and_conclusion(premises, conclusions, coq_script, coq_output_lines=None):
    axioms = set()
    #make phrase
    for conclusion in conclusions:
        premise_preds = []
        if unicodedata.category(conclusion[1]) == "Lo":
            #for Japanese
            matching_premises = get_premises_that_match_conclusion_args(premises, conclusion)
            for premise in matching_premises:
                if re.search("\_.*\s\(?", premise): 
                    premise_preds.append(re.search("(\_.*)\s\(?", premise).group(1))
        else:
            #premise_preds = [premise.split()[2] for premise in matching_premises]
            premise_preds = []
            for premise in premises:
                if re.search("^H[0-9]*", premise):
                    premise_preds.append(premise.split()[2])
        conclusion_pred = conclusion.split()[0]
        pred_args = get_predicate_arguments(premises, conclusion)
        axioms.update(make_axioms_from_preds(premise_preds, conclusion_pred, pred_args, coq_script))
        if not axioms and 'False' not in conclusion_pred:
            failure_log = make_failure_log(
                conclusion_pred, premise_preds, conclusion, premises, coq_output_lines)
            print(json.dumps(failure_log), file=sys.stderr)
    return axioms


def parse_coq_line(coq_line):
    try:
        tree_args = tree_or_string('(' + coq_line + ')')
    except ValueError:
        tree_args = None
    return tree_args


def get_tree_pred_args(line, is_conclusion=False):
    """
    Given the string representation of a premise, where each premise is:
      pX : predicate1 (arg1 arg2 arg3)
      pY : predicate2 arg1
    or the conclusion, which is of the form:
      predicate3 (arg2 arg4)
    returns a tree or a string with the arguments of the predicate.
    """
    tree_args = None
    if not is_conclusion:
        tree_args = parse_coq_line(' '.join(line.split()[2:]))
    else:
        tree_args = parse_coq_line(line)
    if tree_args is None or len(tree_args) < 1:
        return None
    return tree_args[0]


def get_predicate_arguments(premises, conclusion):
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
        args = t.leaves()
        pred_args_list.append([pred] + args)
    conflicting_predicates = set()
    for pa in pred_args_list:
        pred = pa[0]
        args = pa[1:]
        if pred in pred_args and pred_args[pred] != args:
            # check if conflicting predicate candidate is not equal to conclusion
            if pred != conclusion.split()[0]:
                conflicting_predicates.add(pred)
        pred_args[pred] = args
    logging.debug('Conflicting predicates: ' + str(conflicting_predicates))
    for conf_pred in conflicting_predicates:
        del pred_args[conf_pred]
    return pred_args


def make_axioms_from_preds(premise_preds, conclusion_pred, pred_args, coq_script):
    axioms = set()
    linguistic_axioms = \
        get_lexical_relations_from_preds(
            premise_preds, conclusion_pred, coq_script, pred_args)
    axioms.update(set(linguistic_axioms))
    #if not axioms:
    #    approx_axioms = get_approx_relations_from_preds(premise_preds, conclusion_pred, pred_args)
    #    axioms.update(approx_axioms)
    axioms = filter_wrong_axioms(axioms, coq_script)
    return axioms


def get_theorem_line(coq_script_lines):
    for i, line in enumerate(coq_script_lines):
        if line.startswith('Theorem '):
            return i
    assert False, 'There was no theorem defined in the coq script: {0}'\
        .format('\n'.join(coq_script_lines))


def insert_axioms_in_coq_script(axioms, coq_script):
    coq_script_lines = coq_script.split('\n')
    theorem_line = get_theorem_line(coq_script_lines)
    for axiom in axioms:
        axiom_name = axiom.split()[1]
        coq_script_lines.insert(
            theorem_line, 'Hint Resolve {0}.'.format(axiom_name))
        coq_script_lines.insert(theorem_line, axiom)
    new_coq_script = '\n'.join(coq_script_lines)
    return new_coq_script


def try_abductions(coq_scripts):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    while True:
        inference_result_str, direct_proof_scripts, new_direct_axioms = \
            try_abduction(direct_proof_script,
                          previous_axioms=axioms, expected='yes')
        current_axioms = axioms.union(new_direct_axioms)
        reverse_proof_scripts = []
        if not inference_result_str == 'yes':
            inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_abduction(reverse_proof_script,
                              previous_axioms=current_axioms, expected='no')
            current_axioms.update(new_reverse_axioms)
        all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms) or inference_result_str != 'unknown':
            break
        axioms = current_axioms
    return inference_result_str, all_scripts

## for text similarity task
def try_sim_abductions(coq_scripts):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
    all_scripts, direct_proof_scripts, reverse_proof_scripts = [], [], []
    while True:
        #phrasal abduction
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

        if inference_result_str == "unknown":
            #previous abduction
            inference_result_str, direct_proof_scripts, new_direct_axioms = \
            try_abduction(direct_proof_script, previous_axioms=axioms, expected='yes')
            current_axioms = axioms.union(new_direct_axioms)
            if not inference_result_str == 'yes':
                inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_abduction(reverse_proof_script, previous_axioms=current_axioms, expected='no')
                current_axioms.update(new_reverse_axioms)
            all_scripts = direct_proof_scripts + reverse_proof_scripts
        if len(axioms) == len(current_axioms) or inference_result_str != 'unknown':
            break
        axioms = current_axioms

    return inference_result_str, all_scripts

def try_reduce_subgoals(coq_scripts):
    ## detect the number of unprovable subgoals
    ## add admit from coq
    ## return the number of final subgoals and the number of original subgoals
    ## delete assertion
    ##assert len(coq_scripts) == 2
    all_scripts = []
    ## use not negated coq_script
    if len(coq_scripts) == 4:
        direct_proof_script = coq_scripts[-2]
    else:
        direct_proof_script = coq_scripts[-1]
    axioms = set()
    while True:
        inference_result_str, new_direct_proof_script, new_direct_axioms = \
        try_reduce_subgoal(direct_proof_script, previous_axioms=axioms, expected='yes')
        all_scripts.append(new_direct_proof_script)
        if inference_result_str != 'unknown':
            break
    return inference_result_str, all_scripts


def filter_wrong_axioms(axioms, coq_script):
    good_axioms = set()
    for axiom in axioms:
        new_coq_script = insert_axioms_in_coq_script(set([axiom]), coq_script)
        process = Popen(
            new_coq_script,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_lines = [
            line.decode('utf-8').strip().split() for line in process.stdout.readlines()]
        if not is_theorem_error2(output_lines):
            good_axioms.add(axiom)
    return good_axioms


def try_abduction(coq_script, previous_axioms=set(), expected='yes'):
    new_coq_script = insert_axioms_in_coq_script(previous_axioms, coq_script)
    current_tactics = get_tactics()
    debug_tactics = 'repeat nltac_base. try substitution. Qed'
    coq_script_debug = new_coq_script.replace(current_tactics, debug_tactics)
    process = Popen(
        coq_script_debug,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    if is_theorem_defined(l.split() for l in output_lines):
        return expected, [new_coq_script], previous_axioms
    premise_lines = get_premise_lines(output_lines)
    #conclusion = get_conclusion_line(output_lines)
    conclusion = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusion:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_axioms_from_premises_and_conclusion(
        premise_lines, conclusion, coq_script, output_lines)
    axioms = axioms.union(previous_axioms)
    new_coq_script = insert_axioms_in_coq_script(axioms, coq_script)
    process = Popen(
        new_coq_script,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip().split()
                    for line in process.stdout.readlines()]
    inference_result_str = expected if is_theorem_defined(
        output_lines) else 'unknown'
    return inference_result_str, [new_coq_script], axioms


## for text similarity task
def get_subgoal_lines(coq_output_lines):
    ##ConclusionLines
    line_index_last_conclusion_sep = find_final_conclusion_sep_line_index(coq_output_lines)
    if not line_index_last_conclusion_sep:
        return None
    ## extract all subgoals 
    subgoals = []
    subgoalflg = 0
    subgoals.append(coq_output_lines[line_index_last_conclusion_sep+1])
    for line in coq_output_lines[line_index_last_conclusion_sep+1:]:
        if subgoalflg == 1:
            subgoals.append(line)
            subgoalflg = 0
        if re.search("subgoal ", line):
            subgoalflg = 1
    return subgoals


def try_reduce_subgoal(coq_script, previous_axioms=set(), expected='yes'):
    new_coq_script = insert_axioms_in_coq_script(previous_axioms, coq_script)
    current_tactics = get_tactics()
    debug_tactics = 'Set Firstorder Depth 1. nltac. Set Firstorder Depth 3. repeat nltac_base. Qed'
    coq_script_debug = new_coq_script.replace(current_tactics, debug_tactics)

    process = Popen(\
        coq_script_debug, \
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip() for line in process.stdout.readlines()]
    subgoal_lines = get_subgoal_lines(output_lines)
    if not subgoal_lines:
        error_lines = []
        error_flg = 0
        stop_phrases = ["Error: No focused proof (No proof-editing in progress).",\
        "Error: Unknown command of the non proof-editing mode."]
        for o in output_lines:
            if re.search("Error", o) and o not in stop_phrases:
                error_flg = 1
            elif re.search("Error", o) and o in stop_phrases:
                error_flg = 0
            if error_flg == 1:
                error_lines.append(o)
        return 'coq_error,'+"\n".join(error_lines), "", previous_axioms
    return 'yes', coq_script_debug, previous_axioms

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
    premise_lines = get_premise_lines(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    conclusions = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusions:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_phrase_axioms(premise_lines, conclusions, output_lines, expected)
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

def make_phrase_axioms(premises, conclusions, coq_output_lines=None, expected='yes'):
    #check premises and sub-goals, search for their relations from sqlite, select axioms
    axioms = set()
    for conclusion in conclusions:
        matching_premises, conclusion = get_premises_that_partially_match_conclusion_args(premises, conclusion)
        premise_preds = [premise.split()[2] for premise in matching_premises]
        pred_args = get_predicate_case_arguments(matching_premises, conclusion)
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

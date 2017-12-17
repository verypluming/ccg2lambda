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

class AxiomsPhrase(object):
    """
    Create phrasal axioms 
    """
    def __init__(self):
        pass

    def attempt(self, coq_scripts, doc, target, context=None):
        return TryPhraseAbduction(coq_scripts, target)

def TryPhraseAbduction(coq_scripts, target):
    assert len(coq_scripts) == 2
    direct_proof_script = coq_scripts[0]
    reverse_proof_script = coq_scripts[1]
    axioms = set()
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
            #contradiction proof
            inference_result_str, reverse_proof_scripts, new_reverse_axioms = \
                try_phrase_abduction(reverse_proof_script,
                              previous_axioms=axioms, expected='no')
            current_axioms = axioms.union(new_reverse_axioms)
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
    if is_theorem_almost_defined(output_lines):
        return expected, [new_coq_script], previous_axioms
    premise_lines = get_premise_lines(output_lines)
    #for phrase extraction, check all relations between premise_lines and conclusions
    conclusion = get_conclusion_lines(output_lines)
    if not premise_lines or not conclusion:
        failure_log = {"type error": has_type_error(output_lines),
                       "open formula": has_open_formula(output_lines)}
        print(json.dumps(failure_log), file=sys.stderr)
        return 'unknown', [], previous_axioms
    axioms = make_phrase_axioms(premise_lines, conclusion, output_lines, expected)
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
    axioms = set()
    #check sub-goals with normal variables
    conclusions_normal = distinguish_normal_conclusions(conclusions)

    #if existential variables contain in sub-goals, create axioms for sub-goals with existential variables at first
    axioms = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)

    #create axioms for sub-goals with normal variables
    for conclusion in conclusions_normal:
        matching_premises = get_premises_that_partially_match_conclusion_args(premises, conclusion)
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


def is_theorem_almost_defined(output_lines):
    #check if all content subgoals are deleted(remaining relation subgoals can be permitted)
    #ignore relaional subgoals(False, Acc x0=x1) in the proof
    conclusions = get_conclusion_lines(output_lines)
    print("conclusion:{0}".format(conclusions), file=sys.stderr)
    subgoalflg = 0
    if len(conclusions) > 0:
        for conclusion in conclusions:
            if not "False" in conclusion:
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
        return line.split()[2]
    raise(ValueError("Strange coq line: {0}".format(line)))

def check_decomposed(line):
    if line.startswith(':'):
        return False
    if re.search("forall", line):
        return False
    if re.search("exists", line):
        return False
    return True

def make_phrases_from_premises_and_conclusions_ex(premises, conclusions):
    covered_conclusions = set()
    axioms = set()
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
    case_c_preds = [c for c, c_args in c_pred_args.items() if re.search("\?", str(c_args)) and contains_case(str(c_args)) and len(c_args[0]) == 1]
    for case_c_pred in case_c_preds:
        case_c_arg = c_pred_args[case_c_pred][0][0]
        case_c_arg = re.sub(r'\?([0-9]+)', r'x\1', case_c_arg)
        case = re.search(r'([A-Z][a-z][a-z][a-z]?)', case_c_arg).group(1)
        pat = re.compile(case)
        for p_pred, p_args in p_pred_args.items():
            if re.search(pat, p_args[0]):
                #create axioms with premises which have the same case 
                axiom = 'Axiom ax_ex_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                        p_pred,
                        case_c_pred,
                        "x0",
                        "y0")
                axioms.add(axiom)
                #covered_conclusions.add(p)


    #create axioms about sub-goals without case information
    exclude_preds_in_conclusion = {
        get_pred_from_coq_line(l, is_conclusion=True) \
            for l in conclusions if not l.startswith('_') and contains_case(l)}

    for args, c_preds in sorted(c_args_preds.items(), key=lambda x: len(x[0]), reverse=True):
        c_preds = sorted([
            p for p in c_preds if p.startswith('_') and p not in exclude_preds_in_conclusion])
        if len(args) > 1:
            premise_preds = [
                p for p, p_args in p_pred_args.items() if set(p_args).issubset(args)]
            premise_preds = sorted([p for p in premise_preds if not contains_case(p)])
            if premise_preds:
                phrase_pairs.append((premise_preds, c_preds)) # Saved phrase pairs for Yanaka-san.
                for premise_pred in premise_preds:
                #premise_pred = premise_preds[0] #not only the first premise, but all premises are selected
                    for p in c_preds:
                        #if p not in covered_conclusions:
                        c_num_args = max(len(cargs) for cargs in c_pred_args[p])
                        p_num_args = len(p_pred_args[premise_pred])
                        axiom = 'Axiom ax_ex_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
                            premise_pred,
                            p,
                            ' '.join('x' + str(i) for i in range(p_num_args)),
                            ' '.join('y' + str(i) for i in range(c_num_args)))
                        axioms.add(axiom)
                        covered_conclusions.add(p)
    print(axioms) # this is a list of tuples of lists.
    return axioms

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
                        axiom = 'Axiom ax_ex_phrase{0}{1} : forall {2} {3}, {0} {2} -> {1} {3}.'.format(
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

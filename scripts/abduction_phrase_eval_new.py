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
from abduction_phrase_new import *
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

def TryPhraseAbduction(coq_scripts):
    #assert len(coq_scripts) == 2
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
    axioms = make_phrase_axioms(premise_lines, conclusions, output_lines, expected, coq_script_debug)
    #axioms = filter_wrong_axioms(axioms, coq_script) temporarily
    #add only newly generated axioms
    axioms = axioms.union(previous_axioms)

    new_coq_script = insert_axioms_in_coq_script(axioms, coq_script_debug)
    process = Popen(
        new_coq_script,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = [line.decode('utf-8').strip()
                    for line in process.stdout.readlines()]
    inference_result_str = expected if is_theorem_almost_defined(output_lines) else 'unknown'
    return inference_result_str, [new_coq_script], axioms

def make_phrase_axioms(premises, conclusions, coq_output_lines=None, expected='yes', coq_script_debug=None):
    #check premises and sub-goals, search for their relations from sqlite, select axioms
    axioms = set()
    axiom_candidates = make_phrases_from_premises_and_conclusions_ex(premises, conclusions, coq_script_debug, expected)
    axioms = search_axioms_from_db(axiom_candidates)
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

def search_axioms_from_db(axiom_candidates):
    import sqlite3
    axioms = []
    con = sqlite3.connect('./sick_phrase.sqlite3')
    cur = con.cursor()
    phrases = []
    premises_axioms = defaultdict(list)
    for axiom_candidate in axiom_candidates:
        axiomname = axiom_candidate.split(" ")[1]
        premises = axiomname.split("_")[-2:2:-1]
        subgoal = axiomname.split("_")[-1]
        premises_axioms[str(premises)].append(axiom_candidate)
        phrases.append([premises, subgoal])
    for phrase in phrases:
        premises, subgoal = phrase[0], phrase[1]
        for premise in premises:
            #1. search for premise-subgoal relations from sqlite
            df = pd.io.sql.read_sql_query('select * from {table} where premise like \"%{premise}%\" and subgoal = \"{subgoal}\"'\
                .format(table='axioms', premise=premise, subgoal=subgoal), con)
            #select axioms from argument information if there are multiple candidate axioms in database.
            if not df.empty:
                db_premise = df.loc[0, ["premise"]].values[0]
                db_subgoal = df.loc[0, ["subgoal"]].values[0]
                db_kind = df.loc[0, ["kind"]].values[0]
                axioms.extend(premises_axioms[str(premises)])
    con.close()
    return set(axioms)

def _get_phrases(premises, conclusion, expected, coq_script_debug):
    coqaxiom, coqaxioms = "", []
    coq_lists, param_lists, type_lists = [], [], []
    if coq_script_debug:
        coq_lists = coq_script_debug.split("\n")
        param_lists = [re.sub("Parameter ", "", coq_list) for coq_list in coq_lists if re.search("Parameter", coq_list)]
        type_lists = [param_list.split(":") for param_list in param_lists]

    args_id = defaultdict(lambda: len(args_id))
    p_pred_args_id = {}
    p_pred_args = {}
    src_preds = []

    for p in premises:
        predicate = get_pred_from_coq_line(p, is_conclusion=False)
        args = get_tree_pred_args_ex(p, is_conclusion=False)
        if predicate.startswith('_') and args is not None:
            p_pred_args[predicate] = args
            p_pred_args_id[predicate] = [args_id[arg] for arg in args]

    
    conclusion_pred = get_pred_from_coq_line(conclusion, is_conclusion=True)
    if "False" in conclusion_pred or "=" in conclusion_pred:
        #skip relational subgoals
        return []
    conclusion_args = get_tree_pred_args_ex(conclusion, is_conclusion=True)
    c_pred_args_id = [args_id[arg] for arg in conclusion_args]
    trg_pred = denormalize_token(conclusion_pred)
    if contains_case(str(conclusion_args)) is True:
        src_preds = sorted([
                 denormalize_token(prem) for prem, p_args in p_pred_args.items() if set(p_args).issuperset(args) and contains_case(str(p_args))])
    else:
        src_preds = sorted([
                 denormalize_token(prem) for prem, p_args in p_pred_args.items() if set(p_args).issuperset(args) and not contains_case(str(p_args))])

    axioms = search_axioms_from_db(src_preds, trg_pred)
    if axioms:
        for i, axiom in enumerate(axioms):
            if i == 0:
                allargs = []
                premises = [axiom[0] for axiom in axioms]
                for premise in premises:
                    allargs.extend(p_pred_args_id[normalize_token(premise)])
                coqaxiom = 'Axiom ax_phrase_norelation_{0}_{1} : forall {2}, '.format(
                    "_".join(premises),
                    trg_pred,
                    " ".join("x" + str(i) for i in list(set(allargs+c_pred_args_id)))
                    )

            coqaxiom += "_"+axiom[0]+" "+" ".join("x" + str(i) for i in p_pred_args_id[normalize_token(axiom[0])])+" -> "
            if axiom[4] == "antonym" and "Event -> Prop" in check_types(axiom[1], type_lists):
                #antonym axiom for event predicates
                coqaxiom = 'Axiom ax_phrase_antonym_{0}_{1} : forall F x y, {0} x -> _{1} y -> F (Subj x) -> F (Subj y)  -> False.'.format(
                        axiom[0],
                        trg_pred)
                break
            elif axiom[4] == "antonym" and "Entity -> Prop" in check_types(axiom[1], type_lists):
                #antonym axiom for entity->prop predicates
                coqaxiom = 'Axiom ax_phrase_antonym_{0}_{1} : forall F x y, _{0} x -> _{1} y -> F x -> F y  -> False.'.format(
                        axiom[0],
                        trg_pred)
                break
            elif axiom[4] == "antonym":
                #antonym axiom for entity predicates
                coqaxiom = 'Axiom ax_phrase_antonym_{0}_{1} : forall x, _{0} x -> _{1} x -> False.'.format(
                        axiom[0],
                        trg_pred)
                break
        if not re.search("antonym", coqaxiom): 
            coqaxiom += conclusion_pred+" "+" ".join("x" + str(i) for i in c_pred_args_id)+"."
    if coqaxiom:
        coqaxioms.append(coqaxiom)
    return coqaxioms






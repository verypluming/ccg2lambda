# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pascual Martinez-Gomez
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

from collections import defaultdict
import itertools

from linguistic_tools import linguistic_relationship
from linguistic_tools import get_wordnet_cascade
from normalization import denormalize_token, normalize_token
import re


def get_tokens_from_xml_node(node):
    tokens = node.xpath(
        ".//token[not(@base='*')]/@base | //token[@base='*']/@surf")
    return tokens


def get_lexical_relations(doc):
    # Get tokens from all CCG trees and de-normalize them.
    # (e.g. remove the preceding underscore).
    tokens = get_tokens_from_xml_node(doc)
    # For every token pair, extract linguistic relationships.
    relations_to_pairs = defaultdict(list)
    token_pairs = list(itertools.product(tokens, tokens))
    for i, (t1, t2) in enumerate(token_pairs):
        if t1 == t2:
            continue
        # Exclude symmetrical relationships.
        if (t2, t1) in token_pairs[:i]:
            continue
        relations = linguistic_relationship(t1, t2)
        for relation in relations:
            relations_to_pairs[relation].append((t1, t2))
    # For every linguistic relationship, check if 'antonym' is present.
    # If it is present, then create an entry named:
    # Axiom ax_relation_token1_token2 : forall x, _token1 x -> _token2 x ->
    # False.
    antonym_axioms = create_antonym_axioms(relations_to_pairs)
    # Return the axioms as a list.
    axioms = list(itertools.chain(*[antonym_axioms]))
    return list(set(axioms))

def check_types(pred, type_lists):
    for type_list in type_lists:
        if pred in type_list[0]:
            return type_list[1]

def create_antonym_axioms(relations_to_pairs, coq_script_debug=None):
    """
    For every linguistic relationship, check if 'antonym' is present.
    If it is present, then create an entry named:
    Axiom ax_antonym_token1_token2 : forall x, _token1 x -> _token2 x -> False.
    """
    coq_lists, param_lists, type_lists = [], [], []
    if coq_script_debug:
        coq_lists = coq_script_debug.split("\n")
        param_lists = [re.sub("Parameter ", "", coq_list) for coq_list in coq_lists if re.search("Parameter", coq_list)]
        type_lists = [param_list.split(":") for param_list in param_lists]
    relation = 'antonym'
    antonyms = relations_to_pairs[relation]
    axioms = []
    if not antonyms:
        return axioms
    for t1, t2 in antonyms:
        if "Event -> Prop" in check_types(t2.lstrip("_"), type_lists):
        #axiom = 'Axiom ax_{0}_{1}_{2} : forall x, _{1} x -> _{2} x -> False.'\
            axiom = 'Axiom ax_{0}_{1}_{2} : forall F x y, _{1} x -> _{2} y -> F (Subj x) -> F (Subj y)  -> False.'\
                .format(relation, t1, t2)
        else:
            axiom = 'Axiom ax_{0}_{1}_{2} : forall x, _{1} x -> {2} x -> False.'\
                .format(relation, t1, t2)
        axioms.append(axiom)
    return axioms


def create_entail_axioms(relations_to_pairs, relation='synonym'):
    """
    For every linguistic relationship, check if 'relation' is present.
    If it is present, then create an entry named:
    Axiom ax_relation_token1_token2 : forall x, _token1 x -> _token2 x.
    """
    rel_pairs = relations_to_pairs[relation]
    axioms = []
    if not rel_pairs:
        return axioms
    for t1, t2 in rel_pairs:
        axiom = 'Axiom ax_{0}_{1}_{2} : forall x, _{1} x -> _{2} x.'\
                .format(relation, t1, t2)
        axioms.append(axiom)
    return axioms


def create_reventail_axioms(relations_to_pairs, relation='hyponym'):
    """
    For every linguistic relationship, check if 'relation' is present.
    If it is present, then create an entry named:
    Axiom ax_relation_token1_token2 : forall x, _token2 x -> _token1 x.
    **Although the axiom above is correct, there is no influence in the proof.
    **So, temporarily modify the generated axiom below.
    Axiom ax_relation_token1_token2 : forall x, _token1 x -> _token2 x.
    Note how the predicates are reversed.
    """
    rel_pairs = relations_to_pairs[relation]
    axioms = []
    if not rel_pairs:
        return axioms
    for t1, t2 in rel_pairs:
        axiom = 'Axiom ax_{0}_{1}_{2} : forall x, _{1} x -> _{2} x.'\
                .format(relation, t1, t2)
        axioms.append(axiom)
    return axioms


def get_lexical_relations_from_preds(premise_preds, conclusion_pred, pred_args=None, coq_script_debug=None):
    src_preds = [denormalize_token(p) for p in premise_preds]
    trg_pred = denormalize_token(conclusion_pred)

    relations_to_pairs = defaultdict(list)

    for src_pred in src_preds:
        if src_pred == trg_pred or \
           src_pred in '_False' or \
           src_pred in '_True':
            continue
        relations = linguistic_relationship(src_pred, trg_pred)
        # Choose only the highest-priority wordnet relation.
        relation = get_wordnet_cascade(relations)
        relations = [relation] if relation is not None else []
        for relation in relations:
            relations_to_pairs[relation].append((src_pred, trg_pred))

    # TODO: add pred_args into the axiom creation.
    antonym_axioms = create_antonym_axioms(relations_to_pairs, coq_script_debug)
    synonym_axioms = create_entail_axioms(relations_to_pairs, 'synonym')
    hypernym_axioms = create_entail_axioms(relations_to_pairs, 'hypernym')
    similar_axioms = create_entail_axioms(relations_to_pairs, 'similar')
    inflection_axioms = create_entail_axioms(relations_to_pairs, 'inflection')
    derivation_axioms = create_entail_axioms(relations_to_pairs, 'derivation')
    hyponym_axioms = create_reventail_axioms(relations_to_pairs)
    axioms = antonym_axioms + synonym_axioms + hypernym_axioms + hyponym_axioms \
        + similar_axioms + inflection_axioms + derivation_axioms
    return list(set(axioms))

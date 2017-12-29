#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  Copyright 2017 Pascual Martinez-Gomez
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

import unittest

from abduction_phrase import *

class EstimateExistentialVariablesTestCase(unittest.TestCase):
    def test_cut_into_piece(self):
        premises = [
            'H0 : _meat (Acc x1)',
            'H : _lady (Subj x1)',
            'H2 : _cut x1',
            'H6 : _up x1',
            'H5 : _precisely x1',
            'H4 : True',
            'x1 : Event',
            'H3 : True',
            'H1 : True']
        conclusions = [
            '_piece ?775',
            '_into x1 ?775',
            '_woman (Subj x1)']
        #expected_axioms = set([
        #    'Axiom ax_ex_phrase_cut_into : forall x0 y0 y1, _cut x0 -> _into y0 y1.',
        #    'Axiom ax_ex_phrase_cut_piece : forall x0 y0, _cut x0 -> _piece y0.'])
        expected_axioms = set([
            'Axiom ax_ex_phrase_precisely_piece : forall x0 y0, _precisely x0 -> _piece y0.',
            'Axiom ax_ex_phrase_cut_piece : forall x0 y0, _cut x0 -> _piece y0.',
            'Axiom ax_ex_phrase_cut_into : forall x0 y0 y1, _cut x0 -> _into y0 y1.',
            'Axiom ax_ex_phrase_up_into : forall x0 y0 y1, _up x0 -> _into y0 y1.',
            'Axiom ax_ex_phrase_up_piece : forall x0 y0, _up x0 -> _piece y0.',
            'Axiom ax_ex_phrase_precisely_into : forall x0 y0 y1, _precisely x0 -> _into y0 y1.',
            'Axiom ax_ex_phrase_lady_woman : forall x0 y0, _lady x0 -> _woman y0.'
        ])
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_protective_gear(self):
        premises = [
            'H1 : True',
            'x0 : Event',
            'H3 : True',
            'H0 : _equip x0',
            'x1 : Entity',
            'H6 : _with x0 x1',
            'H7 : True',
            'H4 : _protective x1',
            'H8 : _gear x1',
            'H10 : True',
            'H : Acc x0 = Acc x0',
            'H2 : _people (Acc x0)']
        conclusions = [
            '_use ?4844',
            'Acc ?4844 = x1',
            '_protection ?5065',
            '_for ?4844 ?5065']
        #expected_axioms = set([
        #    'Axiom ax_ex_phrase_gear_use : forall x0 y0, _gear x0 -> _use y0.',
        #    'Axiom ax_ex_phrase_gear_protection : forall x0 y0, _gear x0 -> _protection y0.',
        #    'Axiom ax_ex_phrase_gear_for : forall x0 y0 y1, _gear x0 -> _for y0 y1.'])
        expected_axioms = set([
            'Axiom ax_ex_phrase_protective_for : forall x0 y0 y1, _protective x0 -> _for y0 y1.',
            'Axiom ax_ex_phrase_gear_use : forall x0 y0, _gear x0 -> _use y0.',
            'Axiom ax_ex_phrase_gear_for : forall x0 y0 y1, _gear x0 -> _for y0 y1.',
            'Axiom ax_ex_phrase_gear_protection : forall x0 y0, _gear x0 -> _protection y0.',
            'Axiom ax_ex_phrase_protective_use : forall x0 y0, _protective x0 -> _use y0.',
            'Axiom ax_ex_phrase_protective_protection : forall x0 y0, _protective x0 -> _protection y0.'
        ])
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_sidewalk(self):
        premises = [
            'H1 : _skip x1',
            'H2 : _on x1 x2',
            'H3 : _sidewalk x2',
            'H4 : _girl (Subj x1)',
            'H5 : _rope (Acc x1)']
        conclusions = [
            '_street ?2961',
            '_near x1 ?2961']
        expected_axioms = set([
            'Axiom ax_ex_phrase_skip_street : forall x0 y0, _skip x0 -> _street y0.',
            'Axiom ax_ex_phrase_skip_near : forall x0 y0 y1, _skip x0 -> _near y0 y1.'])
        # from pudb import set_trace; set_trace()
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_water_scooter(self):
        #sick_trial_3373 
        #A woman is riding a water scooter.
        #A woman is riding a scooter for water.
        premises = [
            'H1 : _ride x1',
            'H2 : _woman (Subj x1)',
            'H3 : _water (Acc x1)',
            'H4 : _scooter (Acc x1)']
        conclusions = [
            '_for ?2914 (Acc x1)',
            'Subj ?2914 = Acc x1']
        #expected_axioms = set([
        #    'Axiom ax_ex_phrase_scooter_for : forall x0 y0 y1, _scooter x0 -> _for y0 y1.'])
        expected_axioms = set([
        'Axiom ax_ex_phrase_scooter_for : forall x0 y0 y1, _scooter x0 -> _for y0 y1.',
        'Axiom ax_ex_phrase_water_for : forall x0 y0 y1, _water x0 -> _for y0 y1.'
        ])

        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_sewing(self):
        #sick_trial_2166
        #A woman is sewing with a machine.
        #A woman is using a machine made for sewing.
        premises = [
            'H1 : _sew x0',
            'H2 : _with x0 x1',
            'H3 : _machine x1',
            'H4 : _woman (Subj x0)']
        conclusions = [
            '_make ?3353',
            'Acc ?3353 = x1',
            '_sewing ?3565',
            '_for ?3353 ?3565',
            'Acc x0 = x1']
        expected_axioms = set([
            'Axiom ax_ex_phrase_machine_make : forall x0 y0, _machine x0 -> _make y0.',
            'Axiom ax_ex_phrase_machine_for : forall x0 y0 y1, _machine x0 -> _for y0 y1.',
            'Axiom ax_ex_phrase_machine_sewing : forall x0 y0, _machine x0 -> _sewing y0.'])
        # from pudb import set_trace; set_trace()
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_leap(self):
        #sick_trial_6634
        #A hurdle is being leapt by a horse that has a rider on its back.
        #A horse and its rider are leaping over a barrier.
        #containing duplicated sub-goals
        premises = [
            'H1 : _back x4',
            'H2 : _on x5 x4',
            'H3 : _leap x0',
            'H4 : _have x0',
            'H5 : _rider (Subj x5)',
            'H6 : Subj x5 = Acc x0',
            'H7 : _horse (Subj x0)',
            'H8 : _hurdle (Subj x5)']
        conclusions = [
            '_barrier ?5689',
            '_over x0 ?5689',
            '_barrier ?6027',
            '_over x5 ?6027']
        #expected_axioms = set([
        #    'Axiom ax_ex_phrase_have_barrier : forall x0 y0, _have x0 -> _barrier y0.',
        #    'Axiom ax_ex_phrase_have_over : forall x0 y0 y1, _have x0 -> _over y0 y1.'])
        expected_axioms = set([
            'Axiom ax_ex_phrase_leap_over : forall x0 y0 y1, _leap x0 -> _over y0 y1.',
            'Axiom ax_ex_phrase_have_barrier : forall x0 y0, _have x0 -> _barrier y0.',
            'Axiom ax_ex_phrase_have_over : forall x0 y0 y1, _have x0 -> _over y0 y1.',
            'Axiom ax_ex_phrase_leap_barrier : forall x0 y0, _leap x0 -> _barrier y0.'
        ])
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_snowboard_future_work(self):
        #sick_trial_6932 (Future Work)
        #The snowboarder is doing a flip over a mound of snow.
        #Somebody is jumping in the air on a board.
        #There is no argument sharing with sub-goals containing existential variables
        #combine ngram or knowledge base information in searching for related premises
        #in this case, two sub-goal phrases "jump-in-air" and "jump-on-board" are created
        #next, we can detect the sub-goal "board" is related to the premise "snowboarder" using ngram similarity
        #lastly, we may detect the sub-goal "air" is related to the premise "snow" using WordNet
        premises = [
            'H1 : _do x1',
            'H2 : _snow x4',
            'H3 : _mound x4',
            'H4 : _over x3 x4',
            'H5 : _flip (Subj x3)',
            'H6 : Subj x3 = Acc x1',
            'H7 : _snowboarder (Subj x1)']
        conclusions = [
            '_jump ?2675',
            '_air ?2953',
            '_in ?2675 ?2953',
            '_board ?3186',
            '_on ?2675 ?3186']
        expected_axioms = set([
            'Axiom ax_ex_phrase_do_jump : forall x0 y0, _do x0 -> _jump y0.',
            'Axiom ax_ex_phrase_snowboarder_on : forall x0 y0 y1, _snowboarder x0 -> _on y0 y1.',
            'Axiom ax_ex_phrase_snowboarder_board : forall x0 y1, _snowboarder x0 -> _board y1.',
            'Axiom ax_ex_phrase_snow_in : forall x4 y0 y2, _snow x4 -> _in y0 y2.',
            'Axiom ax_ex_phrase_snow_air : forall x4 y2, _snow x4 -> _air y2.'
            ])
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))

    def test_walk(self):
        #sick_train_410 
        #A group of scouts are hiking through the grass.
        #Some people are walking.
        #consider case information in searching for the premises only when a sub-goal also contains case information
        #in this case, solve exiential variable ?1111 by searching for the premise that relates to the sub-goal "_people"
        #then, "_scout" and "_people" have the same case "Subj" and ?1111 can be replaced as x0
        #next, we search for the premise that relates to "_walk x0", and the the premise "_hike x0" can be found
        premises = [
            'H1 : _hike x0',
            'H2 : _through x0 x1',
            'H3 : _grass x1',
            'H4 : _scout (Subj x0)']
        conclusions = [
            '_people (Subj ?1111)',
            '_walk ?1111']
        expected_axioms = set([
            'Axiom ax_ex_phrase_scout_people : forall x0 y0, _scout (Subj x0) -> _people (Subj y0).',
            'Axiom ax_ex_phrase_hike_walk : forall x0 y0, _hike x0 -> _walk y0.'
            ])
        axioms, features = make_phrases_from_premises_and_conclusions_ex(premises, conclusions)
        self.assertEqual(expected_axioms, axioms,
            msg="\n{0} vs. {1}".format(expected_axioms, axioms))


if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(EstimateExistentialVariablesTestCase)
    suites = unittest.TestSuite([suite1])
    unittest.TextTestRunner(verbosity=2).run(suites)

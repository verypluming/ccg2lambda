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

import codecs
import logging
from subprocess import Popen
import subprocess
import time

from nltk2coq import normalize_interpretation
from tactics import get_tactics

class ProofResult(object):
    def __init__(self, judgment, script, output_lines):
        self.judgment = judgment
        self.script = script
        self.output_lines = output_lines

class Theorem(object):
    """
    Defines a theorem that can be piped into a prover.
    It keeps track of the premises, conclusion, execution time,
    number of attempts to prove the theorem, etc.
    """
    def __init__(self, premises, conclusion, library):
        self.premises = premises
        self.conclusion = conclusion
        self.library = library
        self.tactics = get_tactics()
        self.coq_script_direct = self.build_script(premises, conclusion, library)
        negated_conclusion = negate_conclusion(conclusion)
        self.coq_script_reverse = self.build_script(
            premises, negated_conclusion, library)
        self.coq_scripts = []
        # Axioms:
        self.axioms = set()
        self.saturated_axioms_direct = False
        self.saturated_axioms_reverse = False
        # Statistics of execution:
        self.num_execs = 0
        self.execs_time = []

    def build_script(self, premises, conclusion, library):
        formulas = premises + [conclusion]
        coq_formulas = [normalize_interpretation(f) for f in formulas]
        coq_formulas_str = ' -> '.join(coq_formulas)
        coq_script = ('echo \"Require Export coqlib.\n'
            '{0}\nTheorem t1: {1}. {2}.\" | coqtop').format(
            library, coq_formulas_str, self.tactics)
        coq_script = substitute_invalid_chars(coq_script, 'replacement.txt')
        return coq_script

    def get_accumulated_exec_time(self):
        return sum(self.execs_time)

    def get_directional_script(self, expected='yes'):
        coq_script = self.coq_script_direct
        if expected == 'no':
            coq_script = self.coq_script_reverse
        return coq_script

    def is_axiom_saturated(self):
        return self.saturated_axioms_direct and self.saturated_axioms_reverse

    def set_axiom_saturation(self, axioms, expected):
        if self.axioms.union(set(axioms)) == self.axioms:
            if expected == 'yes':
                self.saturated_axioms_direct = True
            if expected == 'no':
                self.saturated_axioms_reverse = True
        else:
            self.saturated_axioms_direct = False
            self.saturated_axioms_reverse = False

    def run(self, axioms, expected='yes', debug=False):
        """
        When we are in debug mode:
        * Replace tactics by debugging tactics to monitor proof.
        * Do not save the script nor the axioms nor the execution times.
        * Do not count the execution as a new execution.
        """
        self.set_axiom_saturation(axioms, expected)
        coq_script = self.get_directional_script(expected)
        augmented_script = InsertAxiomsInCoqScript(axioms, coq_script)
        if debug:
            debug_tactics = 'repeat nltac_base. try substitution. Qed'
            coq_script_debug = augmented_script.replace(self.tactics, debug_tactics)
        else:
            self.coq_scripts.append(augmented_script)
            self.axioms.update(set(axioms))
            t_start = time.time()
        process = Popen(augmented_script, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        output_lines = [
            line.decode('utf-8').strip() for line in process.stdout.readlines()]
        if not debug:
            exec_time = time.time() - t_start
            self.execs_time.append(exec_time)
            self.num_execs += 1
        if is_theorem_defined(l.split() for l in output_lines):
            judgment = expected
        else:
            judgment = None
        result = ProofResult(expected, augmented_script, output_lines)
        return result

# Given a string reprsenting the logical interpretation of the conclusion,
# it returns a string with the negated conclusion.
def negate_conclusion(conclusion):
    return '~(' + conclusion + ')'

# Check whether the string "is defined" appears in the output of coq.
# In that case, we return True. Otherwise, we return False.
def is_theorem_defined(output_lines):
    for output_line in output_lines:
        if len(output_line) > 2 and \
           'is defined' in (' '.join(output_line[-2:])):
            return True
    return False

def substitute_invalid_chars(script, replacement_filename):
    with codecs.open(replacement_filename, 'r', 'utf-8') as finput:
        repl = dict(line.strip().split() for line in finput)
        for invalid_char, valid_char in repl.items():
            script = script.replace(invalid_char, valid_char)
    return script

def InsertAxiomsInCoqScript(axioms, coq_script):
  coq_script_lines = coq_script.split('\n')
  theorem_line = GetTheoremLine(coq_script_lines)
  for axiom in axioms:
    axiom_name = axiom.split()[1]
    coq_script_lines.insert(theorem_line, 'Hint Resolve {0}.'.format(axiom_name))
    coq_script_lines.insert(theorem_line, axiom)
  new_coq_script = '\n'.join(coq_script_lines)
  return new_coq_script

def GetTheoremLine(coq_script_lines):
  for i, line in enumerate(coq_script_lines):
    if line.startswith('Theorem '):
      return i
  assert False, 'There was no theorem defined in the coq script: {0}'\
    .format('\n'.join(coq_script_lines))

#!/usr/bin/python3
# -*- coding: utf-8 -*-
#With the use of txt file (containing Axioms from html results), create txt file for importing sqlite
#future work: implement function for directly putting axiom into sqlite in evaluating training dataset

import re

f = open("./phraseaxiom.txt", "r")
axioms = f.readlines()
f.close()

w = open("./sqlite.txt", "w")
for a in axioms:
    kind = "phrase"
    if re.search("ex_phrase", a):
        kind = "ex_phrase"
        
    allarg = re.search("forall(.*), (.*)", a).group(1)
    allarg = allarg.strip()
    forall = re.search("forall(.*), (.*)", a).group(2)
    
    forall_list = forall.split("->")
    premise = forall_list[0]
    premise_pred = re.search("_([a-z0-9]+)", premise).group(1)
    premise_arg = re.search("_([a-z0-9]+)\s([(xyzSubjAcc0-9)\s]+)", premise).group(2)
    premise_arg = premise_arg.strip()
    subgoal = forall_list[1]
    #print(premise, subgoal)
    subgoal_pred = re.search("_([a-z0-9]+)", subgoal).group(1)
    subgoal_arg = re.search("_([a-z0-9]+)\s([(xyzSubjAcc0-9)\s]+)", subgoal).group(2)
    subgoal_arg = subgoal_arg.strip()
    w.write(premise_pred+"|"+subgoal_pred+"|"+premise_arg+"|"+subgoal_arg+"|"+kind+"|"+allarg+"\n")
w.close()
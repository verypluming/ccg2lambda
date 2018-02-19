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
    premise_preds = []
    premise_args = []
    axiomname = a.split()[1]
    axiomname_list = axiomname.split("_")
    kind = axiomname_list[2]
    subgoal = axiomname_list[-1]
    premise = axiomname_list[-2:2:-1]
        
    #print(";".join(premise), subgoal, kind)
    w.write(";".join(premise)+"|"+subgoal+"|"+kind+"\n")
w.close()

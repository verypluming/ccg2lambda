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
import re
import pandas as pd
g = open("./ablation_new.txt", "r")
commands = g.readlines()
g.close()
for command in commands:
    train, trial = [], []
    command = command.strip()
    lines = command.split("\t")
    name = lines[0]
    if name == "isotype":
        continue
    if re.search("\,", lines[1]):
       	train = [i for i in re.split(r',',lines[1]) if i != '']
        trial = [i for i in re.split(r',',lines[2]) if i != '']
    tb = pd.read_table("./results_20170921_WN/"+name+"_all_result_rte.txt", header=0)
    tp = len(tb.query('entailment_judgment == 1|entailment_judgment == 2&entailment_judgment == correct_answer'))
    tpfp =  len(tb.query('entailment_judgment == 1|entailment_judgment == 2')) 
    prec = float(tp/tpfp)
    tpfn =  len(tb.query('correct_answer == 1|correct_answer == 2')) 
    print(name, tp, tpfp, tpfn)
    rec = float(tp/tpfn)
    corr = len(tb.query('entailment_judgment == correct_answer'))
    acc = float(corr/len(tb))
    w = open("./results/"+name+"_rte_new_result.txt", "w")
    w.write("{:.4f}, {:.4f}, {:.4f}".format(prec, rec, acc))
    w.close()

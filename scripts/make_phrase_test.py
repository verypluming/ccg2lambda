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

from nltk.corpus import wordnet as wn
import difflib
import subprocess
from subprocess import call, Popen
import json
import glob
import re

#extract features for making phrases
#wordnetsim, word2vecsim, ngramsim, argument overlap, rte, similarity

def calc_wordnetsim(sub_pred, prem_pred):
    word_similarity_list = []
    wordFromList1 = wn.synsets(sub_pred)
    wordFromList2 = wn.synsets(prem_pred)
    for w1 in wordFromList1:
        for w2 in wordFromList2:
            if w1.path_similarity(w2) is not None: 
                word_similarity_list.append(w1.path_similarity(w2))
    if(word_similarity_list):
        wordnetsim = max(word_similarity_list)
    else:
        # cannot path similarity but somehow similar
        wordnetsim = 0.5
    return wordnetsim

def calc_word2vecsim(sub_pred, prem_pred):
    process = Popen(\
                'curl http://localhost:5000/word2vec/similarity?w1='+ sub_pred +'\&w2='+ prem_pred, \
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tmp = process.communicate()[0]
    word2vecsim = float(tmp.decode())
    return word2vecsim

def calc_ngramsim(sub_pred, prem_pred):
    ngramsim = difflib.SequenceMatcher(None, sub_pred, prem_pred).ratio()
    return ngramsim

def extract_pred_features(sub_pred, prem_preds):
    prem_pred_cand = {}
    for prem_pred in prem_preds:
        wordnetsim = calc_wordnetsim(sub_pred, prem_pred)
        #word2vecsim = calc_word2vecsim(sub_pred, prem_pred)
        word2vecsim = 0
        ngramsim = calc_ngramsim(sub_pred, prem_pred)
        prem_pred_cand[prem_pred] = [wordnetsim, word2vecsim, ngramsim]
    return prem_pred_cand

def extract_arg_features(sub_arg, prem_infos):
    prem_arg_cand = {}
    
    for prem_info in prem_infos:
        prem_pred = prem_info.split()[2]
        prem_pred = prem_pred.lstrip("_")
        if prem_pred in stopwords:
            continue
        prem_arg = prem_info.split()[3:]
        if str(sub_arg) == str(prem_arg):
            prem_arg_cand[prem_pred] = 1
        else:
            prem_arg_cand[prem_pred] = 0
    return prem_arg_cand

for filename in glob.glob("./results/sick_trial_*.candc.err")[0:4]:
    fileid = re.search("./results/(.*).candc.err", filename).group(1)
    rawfile = open(filename, "r")
    infos = rawfile.readlines()
    rawfile.close()
    
    f = open("./plain/"+fileid+".answer", "r")
    rte = f.readlines()[0].strip()
    f.close()
    if rte == "yes":
        rte_f = [1, 0, 0]
    elif rte == "no":
        rte_f = [0, 1, 0]
    elif rte == "unknown":
        rte_f = [0, 0, 1]
    
    g = open("./plain2/"+fileid+".answer", "r")
    similarity = float(g.readlines()[0].strip())
    g.close()
    
    stopwords = ["Entity", "Event", "True", "False", "Prop"]
    for i in infos:
        info = json.loads(i)
        if "raw sub-goal" not in info:
            continue
        sub_info = info["raw sub-goal"].split()
        if "=" in sub_info:
            #ignore relation sub-goals temporarily
            continue
        sub_arg = sub_info[1:]
        sub_pred = info["unproved sub-goal"]
        if sub_pred in stopwords:
            continue
    
        prem_infos = info["raw premises"]
        prem_preds = info["all premises"]
    
        prem_pred_cand = extract_pred_features(sub_pred, prem_preds)
        prem_arg_cand = extract_arg_features(sub_arg, prem_infos)
            
        for prem_pred in prem_preds:
            features = []
            features.extend(prem_pred_cand[prem_pred])
            features.append(prem_arg_cand[prem_pred])
            features.extend(rte_f)
            features.append(similarity)
            print(sub_pred, prem_pred, features)
    

#example:
#area man [0.14285714285714285, 0, 0.2857142857142857, 0, 1, 0, 0, 4.6]
#area woman [0.1111111111111111, 0, 0.2222222222222222, 0, 1, 0, 0, 4.6]
#area wood [0.125, 0, 0.0, 0, 1, 0, 0, 4.6]
#area through [0.5, 0, 0.18181818181818182, 0, 1, 0, 0, 4.6]
#area walk [0.16666666666666666, 0, 0.25, 0, 1, 0, 0, 4.6]
#area wood [0.125, 0, 0.0, 0, 1, 0, 0, 4.6]
#area through [0.5, 0, 0.18181818181818182, 0, 1, 0, 0, 4.6]
#area walk [0.16666666666666666, 0, 0.25, 0, 1, 0, 0, 4.6]
#wooded man [0.5, 0, 0.0, 0, 1, 0, 0, 4.6]
#wooded woman [0.5, 0, 0.36363636363636365, 0, 1, 0, 0, 4.6]
#wooded wood [0.5, 0, 0.8, 0, 1, 0, 0, 4.6]
#wooded through [0.5, 0, 0.15384615384615385, 0, 1, 0, 0, 4.6]
#wooded walk [0.5, 0, 0.2, 0, 1, 0, 0, 4.6]
#wooded wood [0.5, 0, 0.8, 0, 1, 0, 0, 4.6]
#wooded through [0.5, 0, 0.15384615384615385, 0, 1, 0, 0, 4.6]
#wooded walk [0.5, 0, 0.2, 0, 1, 0, 0, 4.6]
    




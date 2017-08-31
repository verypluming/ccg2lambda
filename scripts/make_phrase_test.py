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
from scipy.spatial import distance

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
        rte_f = 1
    elif rte == "no":
        rte_f = 0.5
    elif rte == "unknown":
        rte_f = 0
    
    g = open("./plain2/"+fileid+".answer", "r")
    similarity = float(g.readlines()[0].strip())
    g.close()
    norm_sim = float((similarity - 1) / (5 - 1))
    
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
            
        best_features = [1,1,1,1,1,1]
        dist = {}
        for prem_pred in prem_preds:
            features = []
            features.extend(prem_pred_cand[prem_pred])
            features.append(prem_arg_cand[prem_pred])
            features.append(rte_f)
            features.append(norm_sim)
            dist[prem_pred] = distance.cityblock(best_features, features)
            print(sub_pred, prem_pred, dist[prem_pred])
        print("final")
        print(sub_pred, min(dist.items(), key=lambda x:x[1])[0], min(dist.values()))
        print("\n")
    

#example:
#area man 3.67142857143
#area woman 3.76666666667
#area wood 2.975
#area through 3.41818181818
#area walk 3.68333333333
#area wood 2.975
#area through 3.41818181818
#area walk 3.68333333333
#final
#area wood 2.975

#wooded man 3.6
#wooded woman 3.23636363636
#wooded wood 1.8
#wooded through 3.44615384615
#wooded walk 3.4
#wooded wood 1.8
#wooded through 3.44615384615
#wooded walk 3.4
#final
#wooded wood 1.8
    




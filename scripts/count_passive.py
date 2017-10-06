#!/usr/bin/python
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
import os
import glob
import numpy as np

from lxml import etree
import re

def get_passive(root):
    passives = 0
    for child in root:
        if re.search("pss", child.get("cat")):
            passives += 1
    return passives

def passive_overlap(sick_id):
    score = 0
    filename = "./parsed/sick_test_"+str(sick_id)+".sem.xml"
    if glob.glob(filename) is None:
        return 0
    else:
        file = glob.glob(filename)
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(file[0], parser)
        t_set = get_passive(tree.xpath("//token[contains(@id, 't0')]"))
        h_set = get_passive(tree.xpath("//token[contains(@id, 't1')]"))
    if t_set or h_set:
        return 1

with open('./results_20170921_WN/all/features_np_again.pickle', 'rb') as in_f:
    train_sources = np.load(in_f)
    train_targets = np.load(in_f)
    #train_sources = np.hstack((train_sources[:, 54:61], train_sources[:, 62:65], train_sources[:, 66:71]))
    #train_sources = train_sources[:, 14:30]
    trial_sources = np.load(in_f)
    #trial_sources = np.hstack((trial_sources[:, 54:61], trial_sources[:, 62:65], trial_sources[:, 66:71]))
    #trial_sources = trial_sources[:, 14:30]
    trial_targets = np.load(in_f)
    train_id = np.load(in_f)
    trial_id = np.load(in_f)

f = open("./results_20170921_WN/passive_list.txt", "w")
for t in trial_id:
    if passive_overlap(t) == 1:
        f.write("{0}".format(t))
        f.write("\n")
f.close()

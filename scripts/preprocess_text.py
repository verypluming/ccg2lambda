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
import json
def format_answer(answer):
    if answer == "entailment":
        return "yes"
    elif answer == "contradiction":
        return "no"
    else:
        return "unknown"

def main():
    f = open("snli_1.0_test.jsonl", "r")
    temp = {i : json.loads(line) for i, line in enumerate(f)}
    i=1
    for k, v in temp.items():
        sentence1 = v["sentence1"]
        sentence2 = v["sentence2"]
        answer = format_answer(v["gold_label"])
        if len(sentence1.split()) < 15 and len(sentence2.split()) < 15:
            g = open("select_snli/sick_test_"+str(i)+".txt", "w")
            g.write(sentence1+"\n")
            g.write(sentence2)
            g.close()
            h = open("select_snli/sick_test_"+str(i)+".answer", "w")
            h.write(answer+"\n")
            h.close()
            i+=1

if __name__ == '__main__':
    main()
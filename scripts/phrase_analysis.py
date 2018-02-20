import re
import os
import sys

#grep -H "Axiom" sick_trial_*.txt.html > fileaxiom.txt
#grep -H "Axiom" sick_train_*.txt.html > fileaxiom.txt
f = open("fileaxiom.txt", "r")
axioms = f.readlines()
f.close()
h = open("formatted_phrases.txt", "w")
oldname = ""
subgoals = []
for axiom in axioms:
    filename = axiom.split(":")[0]
    name = re.search("(.*).txt", filename).group(1)
    rawaxiom = re.split(r"[: ]", axiom)[2]
    subgoal = rawaxiom.split("_")[-1]
    if oldname == name:
        subgoals.append(subgoal)
    else:
        if len(subgoals) > 0:
            h.write(name+"|"+sentences[0].strip()+"|"+sentences[1].strip()+"|"+label.strip()+"|"+"-".join(premises)+"|"+"-".join(subgoals)+"\n")
        oldname = name
        subgoals = []
        subgoals.append(subgoal)
        premises = rawaxiom.split("_")[-2:2:-1]
        g = open("plain/"+name+".txt", "r")
        sentences = g.readlines()
        g.close()
        i = open("plain/"+name+".answer", "r")
        label = i.readlines()[0]
        i.close()

h.close()
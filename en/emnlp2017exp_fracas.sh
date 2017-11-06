#!/bin/bash
# for text similarity task with FraCaS dataset
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

word2vec=$4
if [ "$word2vec" == "word2vec" ]; then
  ./word2vec.sh
fi

# How many processes in parallel you want to run.
# The maximum number should be inferior to the number of cores in your machine.
# Default: 3
cores=${1:-3}
templates=$2

plain_dir=fracas_plain
results_dir=$3

# Usage: 
#
# ./en/emnlp2017exp.sh 3 en/semantic_templates_en_event_sts.yaml (word2vec)
#

# Extract training and test data from FraCaS dataset, removing the header line.
if [ ! -d ${plain_dir} ]; then
  mkdir -p ${plain_dir}
fi


# Copy a coq static library and compile it
cp en/coqlib_sick.v coqlib.v
coqc coqlib.v
# tactics_coq_fracas is better?
cp en/tactics_coq_sick.txt tactics_coq.txt

for f in ./${plain_dir}/fracas_*.txt; do
  ./en/similarity_en_mp_any_fracas.sh $f $templates $results_dir $word2vec;
done &

# Wait for the parallel processes to finish.
wait
 
total=0
correct=0
for f in ./${plain_dir}/fracas_*.answer; do
  let total++
  base_filename=${f##*/}
  sys_filename=./${results_dir}/${base_filename}
  gold_answer=`head -1 $f`
  if [ ! -e ${sys_filename} ]; then
    sys_answer="unknown"
  else
    sys_answer=`head -1 ${sys_filename}`
  fi
  echo -e $f"\t"$gold_answer"\t"$sys_answer
done

# Print a summary (precision, recall, f-score) of the errors at individual problems,
# per problem category and a global score.
echo "Evaluating."
echo "<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Evaluation results of "$category_templates"</title>
  <style>
    body {
      font-size: 1.5em;
    }
  </style>
</head>
<body>
<table border='1'>
<tr>
  <td>fracas problem</td>
  <td>gold answer</td>
  <td>system answer</td>
  <td>proving time</td>
</tr>" > $results_dir/main.html
total_observations=0
correct_recognitions=0
attempts=0
total_proving_time=0
red_color="rgb(255,0,0)"
green_color="rgb(0,255,0)"
white_color="rgb(255,255,255)"
gray_color="rgb(136,136,136)"
for gold_filename in `ls -v ${plain_dir}/fracas_*.answer`; do
  base_filename=${gold_filename##*/} # this line obtains the filename, without the directory path.
  system_filename=${results_dir}/${base_filename/.txt/.answer}
  gold_answer=`cat $gold_filename`
  system_answer=`cat $system_filename`
  time_filename=${results_dir}/${base_filename/.answer/.time}
  proving_time=`cat $time_filename`
  total_proving_time=`echo "$total_proving_time + $proving_time" | bc -l`
  total_number=$((total_number + 1))
  color=$white_color
  if [ "$gold_answer" == "yes" ] || [ "$gold_answer" == "no" ]; then
    total_observations=$((total_observations + 1))
    if [ "$gold_answer" == "$system_answer" ]; then
      correct_recognitions=$((correct_recognitions + 1))
      color=$green_color
    else
      color=$red_color
    fi
    if [ "$system_answer" == "yes" ] || [ "$system_answer" == "no" ]; then
      attempts=$((attempts + 1))
    else
      color=$gray_color
    fi
  fi
  echo '
<tr>
  <td><a style="background-color:'$color';" href="'${base_filename/.answer/.html}'">'${base_filename/.answer/}'</a></td>
  <td>'$gold_answer'</td>
  <td>'$system_answer'</td>
  <td>'$proving_time's</td>
</tr>' >> $results_dir/main.html
done
average_proving_time=`echo "scale=2; $total_proving_time / $total_number" | bc -l`
echo "
<h4><font color="red">Average proving time: "${average_proving_time}" </font></h4>
</body>
</html>
" >> $results_dir/main.html


if [ "$word2vec" == "word2vec" ]; then
  processid=$(ps ax|grep "word2vec-api.py"|grep -v grep|awk '{print $1}')
  kill $processid
fi

python scripts/randomforest_fracas.py --results $results_dir

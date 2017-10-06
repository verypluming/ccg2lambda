#!/bin/sh

nohup python $HOME/new/word2vec-api/word2vec-api.py 2>&1 &
while :
do
  testsim=$(curl http://localhost:5000/word2vec/similarity?w1=person\&w2=woman)
  if [ `echo $testsim| grep '0.547'` ]; then
    echo "word2vec-api started"
    break
  fi
  sleep 100
done

python ./scripts/make_phrase_test.py

#processid=$(ps ax|grep "word2vec-api.py"|grep -v grep|awk '{print $1}')
#kill $processid

#!/bin/sh

nohup python $HOME/word2vec-api/word2vec-api.py 2>&1 &
python ./scripts/make_phrase_test.py

processid=$(ps ax|grep "word2vec-api.py"|grep -v grep|awk '{print $1}')
kill $processid

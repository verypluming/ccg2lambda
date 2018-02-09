# Calculating Japanese semantic textual similarity with ccg2lambda (Beta)
The system for calculating Japanese semantic textual similarity by using features extracted from natural deduction proofs of bidirectional entailment relations between sentence pairs

If you'd like to read README about original ccg2lambda, see [HERE](https://github.com/verypluming/ccg2lambda/tree/japanese_sts/README_original.md)

## Requirement
1. In order to run this system, you need to checkout japanese_sts branch at first:

```bash
git checkout japanese_sts
```

2. You need to download some python modules, Japanese CCG parser(Jigg) and its models
by running the following script:
```bash
./ja/download_dependencies.sh
pip install -r requirements.txt
```
3. Ensure that you have written the parser's location in the files `ja/parser_location_ja.txt`.
```bash
cat ja/parser_location_ja.txt
jigg:/Users/ccg2lambda/ja/jigg-v-0.4
```

4. You also need to download pretrained Word2Vec model([Japanese Wikipedia entity vector](http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/)) and run word2vec-api:
```bash
git clone https://github.com/3Top/word2vec-api
python word2vec-api.py --model entity_vector.model.bin --binary True
```
If official `word2vec-api.py` doesn't work, replace this [word2vec-api.py](https://github.com/verypluming/ccg2lambda/tree/japanese_sts/scripts/word2vec-api.py) 
You can check if `word2vec-api.py` works by:
```bash
curl http://localhost:5000/word2vec/similarity?w1=person\&w2=woman
0.8242756957795436
```

5. Try calculating Japanese semantic textual similarity
## You can try calculating Japanese semantic textual similarity by doing:

```bash
./ja/similarity_ja_mp.sh ja/ja_sts_sample.txt ja/semantic_templates_ja_event.yaml
./ja/similarity_ja_mp.sh ja/ja_sts_sample_2.txt ja/semantic_templates_ja_event.yaml
```

## Output
System output is shown below:
```bash
./ja/similarity_ja_mp.sh ja/ja_sts_sample.txt ja/semantic_templates_ja_event.yaml
jigg parsing ja_plain/ja_sts_sample.txt
semantic parsing ja_parsed/ja_sts_sample.txt.jigg.sem.xml
judging entailment for ja_parsed/ja_sts_sample.txt.jigg.sem.xml 0.653030303030303
./ja/similarity_ja_mp.sh ja/ja_sts_sample2.txt ja/semantic_templates_ja_event.yaml
jigg parsing ja_plain/ja_sts_sample2.txt
semantic parsing ja_parsed/ja_sts_sample2.txt.jigg.sem.xml
judging entailment for ja_parsed/ja_sts_sample2.txt.jigg.sem.xml 0.19848484848484846
```


If you use this software or the semantic templates for your work, please consider citing it.
## A system to compute Semantic Sentence Similarity:

* Hitomi Yanaka, Koji Mineshima, Pascual Martinez-Gomez and Daisuke Bekki. Determining Semantic Textual Similarity using Natural Deduction Proofs. Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, Copenhagen, Denmark, 7-11 September 2017. [arXiv](https://arxiv.org/pdf/1707.08713.pdf)

```
@InProceedings{yanaka-EtAl:2017:EMNLP,
  author    = {Yanaka, Hitomi and Mineshima, Koji  and  Mart\'{i}nez-G\'{o}mez, Pascual  and  Bekki, Daisuke},
  title     = {Determining Semantic Textual Similarity using Natural Deduction Proofs},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
}
```
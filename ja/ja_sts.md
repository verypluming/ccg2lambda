# Learning semantic textual similarity with ccg2lambda
The system for determining semantic textual similarity by combining shallow features with features with features extracted from natural deduction proofs of bidirectional entailment relations between sentence pairs

## Requirement
1. In order to run this system, you need to checkout a different branch at first:

```bash
git checkout emnlp2017_sts
```

2. You need to download some python modules, word2vec-api, the Jigg CCG parser and its models
by running the following script:

```bash
./ja/download_dependencies.sh
git clone https://github.com/3Top/word2vec-api
pip install -r requirements.txt
```

3. Run word2vec-api with your Gensim Word2Vec model(See https://github.com/3Top/word2vec-api in detail)

## Trial with sample text:
You can try extracting features for learning Japanese STS by doing:

```bash
./ja/similarity_ja_mp.sh sample.txt ja/semantic_templates_ja_event.yaml 
```

## Output
System output is shown below:
```bash
features_np.pickle(extracted features from ccg2lambda)
randomforestregressor.pkl(trained model)

results/evaluation.txt(correlation evaluation)
results/error_result.txt(error predictions (diff > 0.75))
results/all_result.txt(all the predictions)
results/result.png(regression line)
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
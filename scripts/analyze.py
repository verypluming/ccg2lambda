import json
import sys
import logging
import random

logging.basicConfig(level=logging.INFO)

fname = sys.argv[1]
entries = []
lines_ignored = 0
lines_without_phrase = 0
with open(fname) as fin:
    for line in fin:
        try:
            data = json.loads(line.strip())
            if len(data.get('phrases', {})) == 0:
                lines_without_phrase += 1
            else:
                entries.append(data)
        except Exception as e:
            lines_ignored += 1

logging.info('Total jsonl entries: {0}'.format(len(entries)))
logging.info('jsonl entries without phrase: {0}'.format(lines_without_phrase))
logging.info('Lines in *.err files ignored: {0}'.format(lines_ignored))

def get_phrases_from_entry(entry):
    phrases_dict = entry.get('phrases', {})
    phrases = []
    for src, trg in phrases_dict.items():
        src_phrase = ' '.join(src.strip('_').split('_'))
        trg_phrase = ' '.join([t.strip('_') for t in trg])
        phrases.append((src_phrase, trg_phrase))
    return phrases

phrases_positive = []
phrases_negative = []
vocabulary = []
for entry in entries:
    phrases = get_phrases_from_entry(entry)
    if entry.get('validity', -1.0) > 0.9:
        phrases_positive.extend(phrases)
    else:
        phrases_negative.extend(phrases)
    for src, trg in phrases:
        vocabulary.extend(src.split())
        vocabulary.extend(trg.split())

phrases_positive = list(set(phrases_positive))
phrases_negative = list(set(phrases_negative))
vocabulary = list(set(vocabulary))

logging.info('Vocabulary size: {0}. Sample: {1}'.format(len(vocabulary), random.sample(vocabulary, 10)))
logging.info('Positive phrases: {0}'.format(len(phrases_positive)))
logging.info('Positive phrases (sample):')
for p in random.sample(phrases_positive, 10):
    logging.info('\t{0} -> {1}'.format(p[0], p[1]))
logging.info('Negative phrases: {0}'.format(len(phrases_negative)))
logging.info('Negative phrases (sample):')
for p in random.sample(phrases_negative, 10):
    logging.info('\t{0} -> {1}'.format(p[0], p[1]))
phrases_pos_neg = set(phrases_positive).intersection(set(phrases_negative))
logging.info('Phrase pairs that are both negative and positive examples: {0}'.format(len(phrases_pos_neg)))
logging.info('Sample:')
for p in random.sample(phrases_pos_neg, 10):
    logging.info('\t{0} -> {1}'.format(p[0], p[1]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from gensim.models import KeyedVectors

text_file = r'/mnt/1B9074BA60C16502/NLP/Pre-Trained Word-Embedding Models/中文 Word2Vec/sgns.merge.300d.word2vec.txt'

logging.basicConfig(
    level=logging.INFO
)
log = logging.getLogger('main')

log.info('加载 Word2Vec 词嵌入预训练输出文件 %r ...', text_file)
word_vectors = KeyedVectors.load_word2vec_format(text_file)
log.info('加载 Word2Vec 词嵌入预训练输出文件 %r 完毕.', text_file)

similarity = word_vectors.similarity('男人', '女人')
print(similarity)

word_vectors.similar_by_word('共党')

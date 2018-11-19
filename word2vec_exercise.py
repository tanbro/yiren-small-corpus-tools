#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import fire
from gensim.models import KeyedVectors


def main(text_file: str):
    logging.basicConfig(
        level=logging.INFO
    )
    log = logging.getLogger('main')
    log.info('加载 Word2Vec 词嵌入预训练输出文件 %r ...', text_file)
    word_vectors = KeyedVectors.load_word2vec_format(text_file)
    log.info('加载 Word2Vec 词嵌入预训练输出文件 %r 完毕.', text_file)

    similarity = word_vectors.similarity('男人', '女人')
    print(similarity)


if __name__ == "__main__":
    fire.Fire(main)

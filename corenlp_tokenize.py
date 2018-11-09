#!/usr/bin/env python
# -*- coding: utf-8 -*-


import fire
import requests
from nltk.parse.corenlp import CoreNLPParser


def tokenize(text, url='http://localhost:9000'):
    """CoreNLP 分词

    Parameters
    ----------
    text : str
        要分词的文本
    url : str, optional
        CoreNLP Web 服务器的 URL (default： 'http://localhost:9000')
    parser : 用于编成接口，不要在命令行使用

    Returns
    -------
    str
        返回空格分隔 token 的句子
    """

    parser = CoreNLPParser(url)
    tokens = list(parser.tokenize(text))
    return ' '.join(tokens)


if __name__ == '__main__':
    fire.Fire(tokenize)

#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
Utility function to generate vocabulary file.
DuReader 词典生成程序

see: https://github.com/baidu/DuReader

改造百度官方的程序

注意这个工具会打乱语料的顺序，除非并发数限制为 1
"""

import argparse
import json
import sys
from itertools import chain

from tqdm import tqdm

__version__ = '2019.1.22b0'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        '--inputs',
        '-i',
        nargs='+',
        type=argparse.FileType('r'),
        required=True,
        help='用于生成词典的 DuReader 语料文件')
    parser.add_argument(
        '--output',
        '-o',
        nargs='?',
        type=argparse.FileType('w'),
        default='-',
        help='输出 DuReader 词典内容到此文件。如果文件已经存在，将清空文件内容 (default=%(default)s)')
    return parser.parse_args()


def main(args):
    """
    Builds vocabulary file from field 'segmented_paragraphs' and 'segmented_question'.
    """
    vocab = {}
    for fno, fp in tqdm(
            enumerate(args.inputs, start=1),
            desc='Read input files',
            total=len(args.inputs)):
        for line in tqdm(fp, desc='Load vocab from file[{}]'.format(fno)):
            obj = json.loads(line.strip())
            paras = [
                chain(*d['segmented_paragraphs']) for d in obj['documents']
            ]
            doc_tokens = chain(*paras)
            question_tokens = obj['segmented_question']
            for t in list(doc_tokens) + question_tokens:
                vocab[t] = vocab.get(t, 0) + 1

    # output
    tqdm.write('Sorting ...')
    sorted_vocab = sorted([(v, c) for v, c in vocab.items()],
                          key=lambda x: x[1],
                          reverse=True)
    for w, c in tqdm(sorted_vocab, desc='Write output file'):
        print('{}\t{}'.format(w, c), file=args.output)


if __name__ == '__main__':
    sys.exit(main(parse_args()))

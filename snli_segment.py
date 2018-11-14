#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 CoreNLP 或者 LTP 对 SNLI/XNLI 中文语料进行分词处理
"""

import argparse
import json
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from random import random
from threading import Lock
from time import sleep
from urllib.parse import urlsplit, urlunsplit

import fire
import requests
from dotenv import load_dotenv
from envs import env
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm

CJK_WHITESPACE_REGEX = re.compile(r'(?P<c>[\u2E80-\u9FFF])(\s+)')


def remove_cjk_whitespace(s):  # type: (str)->str
    """删除字符串中 CJK 文字之间的空格

    :param s: 要处理的字符串

        .. important:: **必须** 是 `UTF-8` 编码，否则工作会不正常

    :return: 删除空格后的字符串
    """
    return re.sub(CJK_WHITESPACE_REGEX, r'\g<c>', s.strip())


def main(input_file='', output_file='', corpus_type='snli', data_format='',
         max_workers=None, flush=True,
         cornlp='', ltp=''):
    """使用 CoreNLP 令牌化 SNLI/XNLI 语料，并输出 SNLI 格式的 JSONL 语料

    Parameters
    ----------

    input_file : str, optional
        输入文件 (default: None, 输入自 stdin)

    output_file : str, optional
        输出文件 (default: None, 输出到 stdout)

    corpus_type : str, optional
        语料格式 "snli" | "xnli" (default: 'snli')

    data_format : str, optional
        文本格式 "jsonl" | "tsv" (default: '', 根据文件名后缀判断)

    max_workers : int, optional
        最大工作进程 (default: None, 根据 CPU 自动分配)

    flush : bool, optional
        输出结果行时是否写缓冲 (default: True)

    cornlp : str, optional
        CoreNLP Web 服务器的 <地址>[:端口]. cornlp 或者 ltp 必须提供其中一个。

    ltp : str, optional
        LTP Web 服务器的 <地址>[:端口]. cornlp 或者 ltp 必须提供其中一个。

    """

    if (not cornlp and not ltp) or (cornlp and ltp):
        print('启动参数错误: cornlp 或者 ltp 必须提供其中一个', file=sys.stderr)
        sys.exit(1)
    if cornlp:
        cornlp = cornlp.strip().strip('/')
    if ltp:
        ltp = ltp.strip().strip('/')
        split_result = urlsplit(ltp, scheme='http', allow_fragments=False)
        if split_result.scheme != 'http':
            print(f'不支持的 URL: {ltp!r}', file=sys.stderr)
            sys.exit(1)
        if split_result.netloc:
            ltp_netloc = split_result.netloc
        else:
            ltp_netloc = split_result.path
        if not ':' in ltp_netloc:
            ltp_netloc += ':12345'
        ltp = urlunsplit([
            split_result.scheme,  # scheme	0	URL scheme specifier	scheme parameter
            ltp_netloc,  # netloc	1	Network location part	empty string
            '/ltp',  # path	2	Hierarchical path	empty string
            '',  # query	3	Query component	empty string
            '',  # fragment	4	Fragment identifier	empty string
        ])

    lines = 0
    if input_file:  # 文本文件
        f_in = open(input_file)
        # 读取行数
        for _ in tqdm(f_in, desc='Counting lines'):
            lines += 1
        f_in.seek(0)
        # 文本格式， jsonl 还是 tsv
        if not data_format:
            file_ext = pathlib.Path(input_file).suffix.lower()
            if file_ext == '.jsonl':
                data_format = 'jsonl'
            elif file_ext == '.tsv':
                data_format = 'tsv'
    else:  # STDIN
        f_in = sys.stdin
    if data_format not in ['jsonl', 'tsv']:
        raise ValueError('无效的 `data_format` 参数')
    # 忽略第一行（标题）
    if input_file and data_format == 'tsv':
        lines -= 1
        f_in.readline()

    if output_file:
        f_out = open(output_file, 'w')
    else:
        f_out = sys.stdout

    # 行处理
    corpus_type = corpus_type.strip().lower()
    lock = Lock()

    # 定义 Executor 中的行处理函数

    def _execute(args):
        index, line = args
        if data_format == 'jsonl':
            d = json.loads(line)
            label = d.get('gold_label', '-').strip().lower()
            sent1 = d['sentence1'].strip()
            sent2 = d['sentence2'].strip()
        elif data_format == 'tsv':
            l = line.split('\t')
            if corpus_type == 'snli':
                label = l[0].strip().lower()
                sent1 = l[5].strip()
                sent2 = l[6].strip()
            elif corpus_type == 'xnli':
                sent1 = l[0].strip()
                sent2 = l[1].strip()
                label = l[2].strip().lower()
            else:
                raise ValueError(f'无效的 `corpus_type`: {corpus_type}')
        else:
            raise ValueError(f'无效的 `data_format`: {data_format}')
        if label == 'contradictory':
            label = 'contradiction'

        if (not sent1) or (not sent2):
            tqdm.write(f'第[{index}]行: 空白语料，将被忽略. {line}', file=sys.stderr)
            return

        sent1 = remove_cjk_whitespace(sent1)
        sent2 = remove_cjk_whitespace(sent2)

        segments = []
        if cornlp:
            parser = CoreNLPParser(cornlp)
            for sent in (sent1, sent2):
                tokens = list(parser.tokenize(sent))
                segments.append(' '.join(tokens))
        elif ltp:
            for sent in (sent1, sent2):
                tokens = []
                r = requests.post(
                    ltp,
                    data={'s': sent, 'x': 'n', 't': 'ws'}
                )
                r.raise_for_status()
                ltp_result = r.json()
                for ltp_sent in ltp_result[0]:
                    for ltp_w in ltp_sent:
                        ws = ltp_w['cont'].strip()
                        if ws:
                            tokens.append(ws)
                segments.append(' '.join(tokens))

        result = json.dumps({
            'index': index,
            'gold_label': label,
            'sentence1': segments[0],
            'sentence2': segments[1],
        }, ensure_ascii=False)

        if output_file:
            with lock:
                print(result, file=f_out, flush=flush)
        else:
            tqdm.write(result)

    # 启动多线程 Executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            for _ in tqdm(
                executor.map(_execute, enumerate(f_in)),
                total=lines, desc='Tokenizing'
            ):
                pass
        except KeyboardInterrupt:
            tqdm.write(f'正在停止...', file=sys.stderr)


if __name__ == '__main__':
    fire.Fire(main)

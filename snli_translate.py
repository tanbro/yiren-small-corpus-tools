#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 有道智云 (http://ai.youdao.com/) 翻译 SNLI/XNLI 语料
"""

import argparse
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from random import random
from threading import Lock
from time import sleep

import fire
import requests
from dotenv import load_dotenv
from envs import env
from tqdm import tqdm

from youdao_translate import TranslateError, translate


def main(input_file=None, output_file=None, corpus_type='snli', data_format='', max_workers=None, max_retry=0, min_retry_sleep=1, max_retry_sleep=10, flush=True, appkey=None, appsecret=None):
    """使用 有道智云 (http://ai.youdao.com/) 翻译 SNLI/XNLI 语料，并输出 SNLI 格式的 JSONL 语料

    Parameters
    ----------

    input_file : str, optional
        输入文件 (default: None, 输入自 stdin

    output_file : str, optional
        输出文件 (default: None, 输出到 stdout)

    corpus_type : str, optional
        语料格式 "snli" | "xnli" (default: 'snli')

    data_format : str, optional
        文本格式 "jsonl" | "tsv" (default: '', 根据文件名后缀判断)

    max_workers : [type], optional
        最大工作进程 (default: None, 根据 CPU 自动分配)

    max_retry : int, optional
        单条语料的翻译失败最大重试次数 (default 0, 不重试)

    min_retry_sleep : int, optional
        单条语料的翻译失败重试最小休眠时间(秒) (default: 1)

    max_retry_sleep : int, optional
        单条语料的翻译失败重试最大休眠时间(秒) (default: 10)

    flush : bool, optional
        输出结果行时是否写缓冲 (default: True)

    appkey : str, optional
        有道翻译 APP Key (default: None, 使用环境变量 YOUDAO_FANYI_APP_KEY)

    appsecret : str, optional
        [description] (default: None, 使用环境变量 YOUDAO_FANYI_APP_SECRET)
    """

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

        tried = 0
        translated = []
        for sent in (sent1, sent2):
            while True:
                try:
                    tried += 1
                    translated.append(translate(
                        sent, appkey=appkey, appsecret=appsecret
                    ))
                except (TranslateError, requests.RequestException) as exception:
                    if max_retry < 0:
                        tqdm.write(
                            f'第[{index}]行: 翻译失败. {exception}', file=sys.stderr)
                        raise
                    else:
                        if tried > max_retry:
                            tqdm.write(
                                f'第[{index}]行: tried={tried} 翻译失败: {line} \n {exception}', file=sys.stderr)
                            raise
                        else:  # sleep a while, retry
                            seconds = float(min_retry_sleep) + \
                                random() * (float(max_retry_sleep) - float(min_retry_sleep))
                            tqdm.write(
                                f'第[{index}]行: tried={tried}, sleep={seconds} 翻译错误: {line} \n {exception}', file=sys.stderr)
                            sleep(seconds)
                            continue
                break

        result = json.dumps({
            'index': index,
            'gold_label': label,
            'sentence1': translated[0],
            'sentence2': translated[1],
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
                total=lines, desc='Translating'
            ):
                pass
        except KeyboardInterrupt:
            tqdm.write(f'正在停止...', file=sys.stderr)


if __name__ == '__main__':
    fire.Fire(main)

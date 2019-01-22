#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用 CoreNLP 对 DuReader 的语料进行分词

see: https://github.com/baidu/DuReader

百度给出的预处理脚本与目前的数据集已经过期不匹配的 -- 缺少其脚本期待的分词后数据。

所以，我们得先进行分词。手头有 CoreNLP 分词。

这个过程有些长，应该把下面的代码放到 DuReader 下运行！

注意这个工具会打乱语料的顺序，除非并发数限制为 1
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import nltk
import requests
from emoji_data import EmojiData
from nltk.parse.corenlp import CoreNLPParser
from opencc import OpenCC
from tqdm import tqdm

__version__ = '2019.01.18b1'

CJK_WHITESPACE_REGEX = re.compile(r'(?P<c>[\u2E80-\u9FFF])(\s+)')

EmojiData.initial()
CC = OpenCC('t2s')  # convert from Traditional Chinese to Simplified Chinese


def remove_cjk_whitespace(s):  # type: (str)->str
    return re.sub(CJK_WHITESPACE_REGEX, r'\g<c>', s.strip())


def pre_segment(s):
    s = s.strip()
    if not s:
        return ''
    # 消除中文之间的空格 - 我们用空格做中文分词！
    s = remove_cjk_whitespace(s)
    if not s:
        return ''
    # CoreNLP 可能因 0xFFFF 以上 Emoji 而崩溃，去掉所有的 Emoji!
    s = EmojiData.get_regex_pattern().sub('', s)
    if not s:
        return ''
    # 全部转换为简体中文
    s = CC.convert(s)
    return s


def segment_one(url, s):
    if nltk.__version__ < '3.4':
        parser = CoreNLPParser(url)
    else:
        parser = CoreNLPParser(url, tagtype='pos')
    return list(parser.tokenize(pre_segment(s)))


def segment_many(url, ss):
    if nltk.__version__ < '3.4':
        parser = CoreNLPParser(url)
    else:
        parser = CoreNLPParser(url, tagtype='pos')
    return [
        list(t) for t in parser.tokenize_sents([pre_segment(s) for s in ss])
    ]


def proc_sample(url, sample_text):
    sample_data = json.loads(sample_text)
    question = sample_data.get('question')
    if question:
        sample_data['segmented_question'] = segment_one(url, question)
    answers = sample_data.get('answers')
    if answers:
        sample_data['segmented_answers'] = segment_many(url, answers)
    for document in sample_data.get('documents', []):
        document['segmented_title'] = segment_one(url, document['title'])
        document['segmented_paragraphs'] = segment_many(
            url, document['paragraphs'])
    return sample_data


def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        '--url',
        '-u',
        type=str,
        default='http://localhost:9000',
        help='Stanford Core NLP Web 服务器的 URL  (default=%(default)s)')
    parser.add_argument(
        '--pool-executor',
        '-p',
        type=str,
        choices=['process', 'thread'],
        default='process',
        help='并发模式 (default=%(default)s)')
    parser.add_argument(
        '--max-workers', '-w', type=int, help='并发数 (default=%(default)s)')
    parser.add_argument(
        '--begin-line',
        '-A',
        type=int,
        default=1,
        help='从语料文件的这一行开始处理 (default=%(default)s)')
    parser.add_argument(
        '--end-line',
        '-B',
        type=int,
        default=0,
        help='到语料文件的这一行停止处理。小于等于0表示不停止处理，直到文件结束 (default=%(default)s)')
    parser.add_argument(
        '--ignore-4xx',
        type=bool,
        default=True,
        help='忽略 Core NLP Web Server 返回的 HTTP 4xx 错误 (default=%(default)s)')
    parser.add_argument(
        '--ignore-5xx',
        type=bool,
        default=True,
        help='忽略 Core NLP Web Server 返回的 HTTP 5xx 错误 (default=%(default)s)')
    parser.add_argument(
        'input',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='要进行分词 DuReader 语料文件 (default=stdin)')
    parser.add_argument(
        'output',
        nargs='?',
        type=argparse.FileType('a'),
        default=sys.stdout,
        help='要输出的具有分词数据的 DuReader 语料文件。如果文件已经存在，将在结尾处另起一行继续输出 (default=stdout)'
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.pool_executor == 'process':
        executor = ProcessPoolExecutor(max_workers=args.max_workers)
    elif args.pool_executor == 'thread':
        executor = ThreadPoolExecutor(max_workers=args.max_workers)
    else:
        raise ValueError(args.pool_executor)

    lines = [line for line in tqdm(args.input)]

    with executor:
        try:
            futs_map = {
                executor.submit(proc_sample, args.url, line_text): (line_no,
                                                                    line_text)
                for line_no, line_text in tqdm(
                    enumerate(lines, start=1), total=len(lines))
                if line_no >= args.begin_line and (
                    args.end_line <= 0 or line_no <= args.end_line)
            }
            prog_bar = tqdm(total=len(lines))
            prog_bar.update(args.begin_line - 1)
            for fut in as_completed(futs_map):
                line_no, line_text = futs_map[fut]
                try:
                    result = fut.result()
                except requests.HTTPError as err:
                    # http status code 5xx
                    if err.response.status_code in range(500, 600):
                        prog_bar.write(
                            '发生分词错误:\n'
                            '    CoreNLP 服务器内部错误: {}\n'
                            '    导致错误的样本: {}行'.format(err, line_no),
                            file=sys.stderr)
                        if not args.ignore_5xx:
                            raise
                    # http status code 5xx
                    elif err.response.status_code in range(400, 500):
                        prog_bar.write(
                            '发生分词错误:\n'
                            '    CoreNLP 服务器收到的请求有误: {}\n'
                            '    导致错误的样本: {}行'.format(err, line_no),
                            file=sys.stderr)
                        if not args.ignore_4xx:
                            raise
                    else:  # 其它 http 错误码
                        prog_bar.write(
                            '发生分词错误，任务异常中止:\n'
                            '    CoreNLP 服务器返回错误: {}\n'
                            '    导致错误的样本: {}行\n'
                            '{}'.format(err, line_no, line_text),
                            file=sys.stderr)
                        raise
                except KeyboardInterrupt:
                    raise  # re-raise
                except Exception as err:  # 其它异常
                    prog_bar.write(
                        '分词的执行出现错误，任务异常中止: {}\n'
                        '    导致错误的样本: {}行\n'
                        '{}'.format(err, line_no, line_text),
                        file=sys.stderr)
                    raise
                else:
                    print(
                        json.dumps(result, ensure_ascii=False),
                        file=args.output)
                    prog_bar.update()

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    sys.exit(main(opts()))

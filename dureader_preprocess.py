#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DuReader 预处程序

see: https://github.com/baidu/DuReader

改造百度官方的预处理程序

注意这个工具会打乱语料的顺序，除非并发数限制为 1
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

__version__ = '2019.1.21b0'


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def find_best_question_match(doc, question, with_score=False):
    """
    For each document, find the paragraph that matches best to the question.
    Args:
        doc: The document object.
        question: The question tokens.
        with_score: If True then the match score will be returned,
            otherwise False.
    Returns:
        The index of the best match paragraph, if with_score=False,
        otherwise returns a tuple of the index of the best match paragraph
        and the match score of that paragraph.
    """
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
        if len(question) > 0:
            related_score = metric_max_over_ground_truths(
                recall, para_tokens, question)
        else:
            related_score = 0

        if related_score > max_related_score \
                or (related_score == max_related_score
                    and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    if most_related_para == -1:
        most_related_para = 0
    if with_score:
        return most_related_para, max_related_score
    return most_related_para


def find_fake_answer(sample):
    """
    For each document, finds the most related paragraph based on recall,
    then finds a span that maximize the f1_score compared with the gold answers
    and uses this span as a fake answer span
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    for doc in sample['documents']:
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(sample['segmented_answers']) > 0:
                related_score = metric_max_over_ground_truths(
                    recall, para_tokens, sample['segmented_answers'])
            else:
                continue
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
        doc['most_related_para'] = most_related_para

    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set(
            [token for token in segmented_answer])
    for d_idx, doc in enumerate(sample['documents']):
        if not doc['is_selected']:
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][
            doc['most_related_para']][:1000]
        for start_tidx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue
            for end_tidx in range(
                    len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx:end_tidx + 1]
                if len(sample['segmented_answers']) > 0:
                    match_score = metric_max_over_ground_truths(
                        f1_score, span_tokens, sample['segmented_answers'])
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_d_idx = d_idx
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
    if best_match_score > 0:
        sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)

    return sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        '--max-workers', '-w', type=int, help='并发数 (default=%(default)s)')
    parser.add_argument(
        'input',
        nargs='?',
        type=argparse.FileType('r'),
        default='data/raw/trainset/search.train.json',
        help='要进行预处理 DuReader 语料文件 (default=%(default)s)')
    parser.add_argument(
        'output',
        nargs='?',
        type=argparse.FileType('a'),
        default='data/preprocessed/trainset/search.train.json',
        help=
        '要输出的预处理后的 DuReader 语料文件。如果文件已经存在，将在结尾处另起一行继续输出 (default=%(default)s)')
    return parser.parse_args()


def main(args):
    print('load:')
    samples = []
    for index, line in enumerate(args.input):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.decoder.JSONDecodeError as err:
            tqdm.write(
                '行[{}] JSON 解码错误，该样本将被忽略。\n  错误：n{}'.format(index + 1, err),
                file=sys.stderr)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        print('submit:')
        fut_list = [
            executor.submit(find_fake_answer, sample)
            for sample in tqdm(samples)
        ]

        samples = []  # release

        print('wait:')
        for fut in tqdm(as_completed(fut_list), total=len(fut_list)):
            txt = json.dumps(fut.result(), ensure_ascii=False)
            print(txt, file=args.output)


if __name__ == '__main__':
    sys.exit(main(parse_args()))

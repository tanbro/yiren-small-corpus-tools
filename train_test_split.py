#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将多行语料文本文件划分成训练和测试两个部分
"""

from typing import Optional, Union

import fire
from sklearn.model_selection import train_test_split

Number = Optional[Union[int, float]]


def main(input_file: str, train_file: str, test_file: str, test_size: Number = 0.25, train_size: Number = None, random_state: Optional[int] = None, shuffle: bool = True):
    """Split "A sample per line" corpus text file into random train and test subsets

    Parameters
    ----------

    input_file : str
        Path of the "A sample per line" corpus text file

    train_file : str
        Path of the file to store splitted train data. 

    test_file : str
        Path of the file to store splitted test data

    test_size : float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. 
        By default, the value is set to 0.25.
        It will remain 0.25 only if ``train_size`` is unspecified, otherwise it will complement the specified ``train_size``.

    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting.
    """

    with open(input_file) as fd_in:
        ds_train, ds_test = train_test_split(
            fd_in.readlines(),
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    with open(train_file, 'w') as fd_train, open(test_file, 'w') as fd_test:
        fd_train.writelines(ds_train)
        fd_test.writelines(ds_test)


if __name__ == '__main__':
    fire.Fire(main)

"""
A Pylearn2 Dataset object for accessing the data for the
Kaggle contest for the IFT 6266 H13 course.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import csv
import numpy as np

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class ContestDataset(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing the data for the
    Kaggle contest for the IFT 6266 H13 course.
    """

    def __init__(self, which_set,
            base_path = '/data/lisatmp/ift6266h13/ContestDataset'):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        """

        files = {'train': 'train.csv', 'public_test' : 'test.csv'}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        path = base_path + '/' + filename

        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        row = reader.next()

        y_list = []
        X_list = []

        for row in reader:
            y_str, X_row_str = row
            y = int(y_str)
            y_list.append(y)
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list)
        y = np.asarray(y_list)

        view_converter = DefaultViewConverter(shape=[48,48,1])

        super(ContestDataset, self).__init__(X=X, y=y, view_converter=view_converter)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

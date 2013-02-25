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
            base_path = '/data/lisatmp/ift6266h13/ContestDataset',
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),
            fit_test_preprocessor = False):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.test_args = locals()
        self.test_args['which_set'] = 'public_test'
        self.test_args['fit_preprocessor'] = fit_test_preprocessor
        del self.test_args['start']
        del self.test_args['stop']
        del self.test_args['self']

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
            if which_set == 'train':
                y_str, X_row_str = row
                y = int(y_str)
                y_list.append(y)
            else:
                X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list)
        if which_set == 'train':
            y = np.asarray(y_list)

            one_hot = np.zeros((y.shape[0],7),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot
        else:
            y = None

        if start is not None:
            assert which_set != 'test'
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            if y is not None:
                y = y[start:stop, :]

        view_converter = DefaultViewConverter(shape=[48,48,1], axes=axes)

        super(ContestDataset, self).__init__(X=X, y=y, view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

    def get_test_set(self):
        return ContestDataset(**self.test_args)


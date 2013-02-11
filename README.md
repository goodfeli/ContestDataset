ContestDataset
==============

A Pylearn2 Dataset object for accessing the dataset used for the kaggle competition of IFT 6266 H13

To use this functionality in python, add this directory to your PYTHONPATH
environment variable. You can now run

import contest_dataset

This directory also contains two yaml files describing the train and the
public test set for the contest. You can view this using the pylearn2
show_examples.py script. Add pylearn2/scripts to your PATH variable.
You can now run

show_examples.py train.yaml

or

show_examples.py public_test.yaml

to visualize a few examples from either dataset.

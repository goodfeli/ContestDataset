__author__ = 'Vincent Archambault-Bouffard'
__credits__ = ['Ian Goodfellow', 'Vincent Archambault-Bouffard']

import sys
import numpy as np
import csv
from theano import function


def usage():
    print """usage: python make_submission.py model.pkl submission.csv
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will make submission.csv, which you may then upload to the
kaggle site."""


if len(sys.argv) != 3:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, out_path = sys.argv

import os

if os.path.exists(out_path):
    usage()
    print out_path + " already exists, and I don't want to overwrite anything just to be safe."
    quit(-1)

from pylearn2.utils import serial

try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()

# use smallish batches to avoid running out of memory
batch_size = 100
model.set_batch_size(batch_size)
# dataset must be multiple of batch size of some batches will have
# different sizes. theano convolution requires a hard-coded batch size
m = dataset.X.shape[0]
extra = batch_size - m % batch_size
assert (m + extra) % batch_size == 0
if extra > 0:
    dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
                                                    dtype=dataset.X.dtype)), axis=0)
assert dataset.X.shape[0] % batch_size == 0

X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)
f = function([X], Y)

y = []

for imgIdx in xrange(dataset.X.shape[0] / batch_size):
    x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
    if X.ndim > 2:
        x_arg = dataset.get_topological_view(x_arg)
    y.append(f(x_arg.astype(X.dtype)))

y = np.concatenate(y)
assert y.shape[0] == dataset.X.shape[0]
# discard any zero-padding that was used to give the batches uniform size
y = y[:m]

submission = []
with open('submissionFileFormat.csv', 'rb') as cvsTemplate:
    reader = csv.reader(cvsTemplate)
    for row in reader:
        submission.append(row)

idx = 1
for row in submission:
    imgIdx = row[1] - 1
    keypointIndex = row[2]


for imgIdx in range(y.shape[0]):
    for j in range(30):
        if submission[idx][1] != imgIdx: # We need to go to the next image
            break
        if y[imgIdx][j] != -1:
            submission[idx].append(y[imgIdx][j])
            idx += 1

with open(out_path, 'w') as cvsTemplate:
    writer = csv.writer(cvsTemplate)
    for row in submission:
        writer.writerow(row)
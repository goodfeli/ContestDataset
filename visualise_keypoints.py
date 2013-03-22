__author__ = 'Vincent Archambault-Bouffard'
__credits__ = ['Ian Goodfellow', 'Vincent-Archambault-Bouffard']

import sys
import os
from pylearn2.config import yaml_parse
from theano import function
from PIL import Image, ImageDraw
import numpy as np


def drawKeypointsOnImage(img, keyPoints):
    """
    Returns an RGB image with the keypoints added to it.
    Green for left side and red for right side. (relative to subject)
    Original author = Pierre-Luc Carrier
    """

    cp = img.copy().convert("RGB")

    draw = ImageDraw.Draw(cp)
    draw.setink("#00ff00")

    leftFill = (0, 255, 0)
    rightFill = (255, 0, 0)

    left_eye_center_x = 0
    left_eye_inner_corner_x = 4
    left_eye_outer_corner_x = 6
    left_eyebrow_inner_end_x = 12
    left_eyebrow_outer_end_x = 14
    mouth_left_corner_x = 22

    for i in range(len(keyPoints) / 2):
        if keyPoints[i * 2] is not None and keyPoints[i * 2 + 1] is not None:
            if i * 2 in [left_eye_center_x,
                         left_eye_inner_corner_x,
                         left_eye_outer_corner_x,
                         left_eyebrow_inner_end_x,
                         left_eyebrow_outer_end_x,
                         mouth_left_corner_x,
                         left_eye_center_x]:
                fill = leftFill
            else:
                fill = rightFill
            draw.ellipse((int(keyPoints[i * 2]), int(keyPoints[i * 2 + 1]),
                          int(keyPoints[i * 2]) + 4, int(keyPoints[i * 2 + 1]) + 4),
                         fill=fill)

    del draw
    return cp


def usage():
    print """usage: python visualise_keypoints.py model.pkl folderName
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will create a new directory called "folderName" and generate
images with keypoints computed by the model"""


if len(sys.argv) != 3:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, out_path = sys.argv

if os.path.exists(out_path):
    usage()
    print out_path + " already exists, and I don't want to overwrite anything just to be safe."
    quit(-1)
os.makedirs(out_path)

from pylearn2.utils import serial
try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

# Loads model and dataset
dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()

# Use smallish batches to avoid running out of memory
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

# Computes Y values
X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)
f = function([X], Y)
y = []

for i in xrange(dataset.X.shape[0] / batch_size):
    x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
    if X.ndim > 2:
        x_arg = dataset.get_topological_view(x_arg)
    y.append(f(x_arg.astype(X.dtype)))

y = np.concatenate(y)
assert y.shape[0] == dataset.X.shape[0]
# discard any zero-padding that was used to give the batches uniform size
y = y[:m]


# For each image we superpose the key points
images = dataset.get_topological_view()
images = dataset.adjust_for_viewer(images)
for i, kp in enumerate(y):
    if i % 100 == 0:
        print "Saving image {0}".format(i)

    image = images[i]
    # Convert image back to 0 .. 255 values
    image *= 0.5
    image += 0.5
    image *= 255
    # Transform to PIL image
    image = np.cast['uint8'](image)
    image = Image.fromarray(image[:, :, 0])
    # Superpose key points
    image = drawKeypointsOnImage(image, kp)
    # Save to file
    savePath = os.path.join(out_path, "{0}.png".format(i))
    image.save(savePath)

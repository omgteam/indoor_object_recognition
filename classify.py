import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import webbrowser
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from PIL import Image
from my_utils import file_2_array
from parameters import *

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

n_in = ishape[0]*ishape[1]
n_out = log_n_out

f = open(lenet_params.save)
best_params = cPickle.load(f)
f.close()

rng = numpy.random.RandomState(23455)
# classify
x = T.dmatrix('x')
y = T.lvector('y')


layer0_input = x.reshape((1, 3, ishape[0], ishape[1]))
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
        image_shape=(1, 3, ishape[0], ishape[1]),
        filter_shape=(nkerns[0], 3, 5, 5), poolsize=(4, 4))
layer0.W.set_value(best_params[8].get_value(), borrow=True)
layer0.b.set_value(best_params[9].get_value(), borrow=True)

layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(1, nkerns[0], 19, 19),
        filter_shape=(nkerns[1], nkerns[0], 6, 6), poolsize=(2, 2))
layer1.W.set_value(best_params[6].get_value(), borrow=True)
layer1.b.set_value(best_params[7].get_value(), borrow=True)

layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
        image_shape=(1, nkerns[1], 7, 7),
        filter_shape=(nkerns[2], nkerns[1], 6, 6), poolsize=(1, 1))
layer2.W.set_value(best_params[4].get_value(), borrow=True)
layer2.b.set_value(best_params[5].get_value(), borrow=True)

layer3_input = layer2.output.flatten(2)
layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[2] * 2 * 2,
                     n_out=500, activation=T.tanh)
layer3.W.set_value(best_params[2].get_value(), borrow=True)
layer3.b.set_value(best_params[3].get_value(), borrow=True)

layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=13)
layer4.W.set_value(best_params[0].get_value(), borrow=True)
layer4.b.set_value(best_params[1].get_value(), borrow=True)

# compiled theano function that returns this value
classify = theano.function(inputs=[layer0_input], outputs=layer4.y_pred)
classes = ['camera', 'ceiling_fan', 'cellphone', 'chair', 'cup', \
        'faces', 'lamp', 'laptop', 'revolver', 'scissors', 'stapler', \
        'umbrella', 'watch']

def top5(path):
    get_p_y_given_x = theano.function(inputs=[layer0_input], outputs=layer4.p_y_given_x)
    plist = list(get_p_y_given_x(file_2_array(path, ishape[0], ishape[1]))[0])
    plist = list(enumerate(plist))
    plist.sort(cmp=lambda x, y: cmp(x[1], y[1]))
    rt = []
    for i in xrange(5):
        tmp = plist.pop()
        rt.append((classes[tmp[0]], format(tmp[1], '.2%')))
    return rt

def classify_file(path):
    return classes[classify(file_2_array(path, ishape[0], ishape[1]))[0]]

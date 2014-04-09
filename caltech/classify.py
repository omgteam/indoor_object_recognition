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

if __name__ == '__main__':
    n_in = 32 * 32
    n_out = 4
    # load data and get test dataset
    dataset = 'caltech_small.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # load data into shared variables
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    f = open('lenet_params1.save')
    best_params = cPickle.load(f)
    f.close()

    rng = numpy.random.RandomState(23455)
    # classify
    x = T.dmatrix('x')
    y = T.lvector('y')
    nkerns=[20, 50]

    layer0_input = x.reshape((1, 3, 32, 32))
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(1, 3, 32, 32),
            filter_shape=(nkerns[0], 3, 5, 5), poolsize=(2, 2))
    layer0.W.set_value(best_params[6].get_value(), borrow=True)
    layer0.b.set_value(best_params[7].get_value(), borrow=True)

    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(1, nkerns[0], 14, 14),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
    layer1.W.set_value(best_params[4].get_value(), borrow=True)
    layer1.b.set_value(best_params[5].get_value(), borrow=True)

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 5 * 5,
                         n_out=200, activation=T.tanh)
    layer2.W.set_value(best_params[2].get_value(), borrow=True)
    layer2.b.set_value(best_params[3].get_value(), borrow=True)

    layer3 = LogisticRegression(input=layer2.output, n_in=200, n_out=4)
    layer3.W.set_value(best_params[0].get_value(), borrow=True)
    layer3.b.set_value(best_params[1].get_value(), borrow=True)

    x_value = test_set_x.get_value()
    x_values = []
    for val in x_value:
        x_values.append(val.reshape((1, 3, 32, 32)))
    # compiled theano function that returns this value
    classify = theano.function(inputs=[layer0_input], outputs=layer3.y_pred)
    get_p_y_given_x = theano.function(inputs=[layer0_input], outputs=layer3.p_y_given_x)

    result = []
    for x in x_value:
        tmp = x.reshape(1, 3, 32, 32)
        result.append(classify(tmp)[0])
    print 'the result is...'
    print result
    print 'the answer is...'
    print test_set_y.eval()[0:40]

    classes = ['cup', 'camera', 'cellphone', 'chair']

    def classify_file(path):
        return classes[classify(file_2_array(path))[0]]

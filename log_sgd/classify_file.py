import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import webbrowser
from PIL import Image
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
    def classify(self, input):
         theano.function(inputs=None, outputs=y_pred)
    def get_result(self, data):
        x = T.dmatrix('x')
        p_y = T.nnet.softmax(T.dot(x, self.W) + self.b)
        get_p_y = theano.function(inputs=[x], outputs=p_y)
        return get_p_y(data)
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    n_in = 3 * 150 * 150
    n_out = 4

    print 'loading W and b...'
    # load weights and bias
    W = theano.shared(value=numpy.zeros((n_in, n_out)), name='W')
    b = theano.shared(value=numpy.zeros((n_out,)), name='b')
    f = open('logistic_caltech.save')
    W.set_value(cPickle.load(f), borrow=True)
    b.set_value(cPickle.load(f), borrow=True)
    f.close()

    # classify
    x = T.dmatrix('x')
    y = T.lvector('y')

    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

    get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)

    # symbolic description of how to compute prediction as class whose probability
    # is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)

    # compiled theano function that returns this value
    classify = theano.function(inputs=[x], outputs=y_pred)

    classes = ['cup', 'camera', 'cellphone', 'chair']
    while True:
        arg = raw_input('image number:')
        if arg == '':
            continue
        img = Image.open(open('images/' + str(arg) + '.jpg'))
        img = numpy.asarray(img, dtype='float64') / 256.
        img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3 * 150 * 150)
        result = classify(img_)
        print(classes[result])

    #for i in xrange(10):
     #   cl = classify(x_value[i:i+1])
      #  print str(cl[0]) + ': ' + str(get_p_y_given_x(x_value)[i][cl[0]])

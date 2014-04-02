import cPickle
import numpy
import gzip
import pylab
import webbrowser
import theano.tensor as T
from theano import *
from PIL import Image
from theano.tensor.nnet import conv

if __name__ == '__main__':
  dataset = 'mnist.pkl.gz'
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()
  for num in xrange(10):
	  img = Image.new('L', (28, 28))
	  list = []
	  for x in xrange(28):
	  	for y in xrange(28):
	  		val = float(test_set[0][num][x * 28 + y])
	  		val *= 255
	   		list.append(val)
	  img.putdata(list)
	  img.save('mnist/' + str(num) + '.jpg')
  
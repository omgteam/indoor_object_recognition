import gzip
import cPickle
import pylab
import numpy
import theano.tensor as T
import math
from theano import *
from theano.tensor.nnet import conv
from PIL import Image

def load_weights(path):
    W = theano.shared(value=numpy.zeros((n_in, n_out)), name='W')
    b = theano.shared(value=numpy.zeros((n_out,)), name='b')
    f = open(path)
    W.set_value(cPickle.load(f), borrow=True)
    b.set_value(cPickle.load(f), borrow=True)
    f.close()
    return W, b

def gen_img(arr, path):
  l = int(math.sqrt(len(arr)))
  img = Image.new('L', (l, l))
  list = []
  for x in xrange(l):
  	for y in xrange(l):
  		val = float(arr[x * l + y])
  		val *= 255
   		list.append(val)
  img.putdata(list)
  img.save(path)

def gen_rn_weights():
  rng = numpy.random.RandomState(23455)
  input = T.tensor4(name='input')
  w_shp = (2, 3, 9, 9)
  w_bound = numpy.sqrt(3 * 9 * 9)
  W = theano.shared( numpy.asarray(
              rng.uniform(
                  low=-1.0 / w_bound,
                  high=1.0 / w_bound,
                  size=w_shp),
              dtype=input.dtype), name ='W')
  b_shp = (2,)
  b = theano.shared(numpy.asarray(
              rng.uniform(low=-.5, high=.5, size=b_shp),
              dtype=input.dtype), name ='b')
  return W, b

def make_dataset(pkl_path):
  x_list, y_list = [], []
  classes = ['cup', 'camera', 'cellphone', 'chair']
  for i in xrange(40):
    for cl in xrange(len(classes)):
      img_str = 'image_' + (4 - len(str(i + 1))) * '0' + str(i + 1) + '.jpg'
      img = Image.open(open('images/' + classes[cl] + '/' + img_str))
      img = numpy.asarray(img, dtype='float64') / 256.
      img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(3 * 150 * 150)
      x_list.append(img)
      y_list.append(cl)
  train_set = tuple([x_list, y_list])
  x_list, y_list = [], []
  for i in xrange(10):
    for cl in xrange(len(classes)):
      img_str = 'image_' + (4 - len(str(i + 41))) * '0' + str(i + 41) + '.jpg'
      img = Image.open(open('images/' + classes[cl] + '/' + img_str))
      img = numpy.asarray(img, dtype='float64') / 256.
      img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(3 * 150 * 150)
      x_list.append(img)
      y_list.append(cl)
  test_set = tuple([x_list, y_list])
  valid_set = tuple([x_list, y_list])

  # saving
  f = file(pkl_path, 'wb')
  cPickle.dump([train_set, test_set, valid_set], f, -1)
  f.close()
  from subprocess import check_call
  check_call('gzip -f ' + pkl_path,shell=True)  

if __name__ == '__main__':
  make_dataset('data/caltech.pkl')

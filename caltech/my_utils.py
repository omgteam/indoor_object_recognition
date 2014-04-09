import gzip
import cPickle
import pylab
import numpy
import theano.tensor as T
import math
from theano import *
from theano.tensor.nnet import conv
from PIL import Image

def file_2_array(path):
  img = Image.open(open(path))
  img = numpy.asarray(img, dtype='float64') / 256.
  img = img.swapaxes(0,2).swapaxes(1,2).reshape(1, 3, 32, 32)
  return img

def small_dataset(pkl_path):
  x_list, y_list = [], []
  classes = ['cup', 'camera', 'cellphone', 'chair']
  for i in xrange(110):
    for cl in xrange(len(classes)):
      img_str = 'image (' + str(i + 1) + ')' + '.jpg'
      img = Image.open(open('small/' + classes[cl] + '/' + img_str))
      img = numpy.asarray(img, dtype='float64') / 256.
      img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(3 * 32 * 32)
      x_list.append(img)
      y_list.append(cl)
  train_set = tuple([x_list, y_list])
  x_list, y_list = [], []
  for i in xrange(10):
    for cl in xrange(len(classes)):
      img_str = 'image (' + str(i + 101) + ')' + '.jpg'
      img = Image.open(open('small/' + classes[cl] + '/' + img_str))
      img = numpy.asarray(img, dtype='float64') / 256.
      img = img.swapaxes(0, 2).swapaxes(1, 2).reshape(3 * 32 * 32)
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
  small_dataset('caltech_small.pkl')

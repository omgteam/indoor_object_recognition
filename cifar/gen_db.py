import gzip
import cPickle
import pylab
import numpy
import math
from logistic_sgd import load_data
from PIL import Image

def gen_img(arr, path):
  l = len(arr) / 3
  l = int(math.sqrt(l))
  img = Image.new('RGB', (l, l))
  list = []
  for x in xrange(l):
    for y in xrange(l):
      r = int(arr[x * l + y])
      g = int(arr[x * l + y + 1024])
      b = int(arr[x * l + y + 2048])
      list.append((r, g, b))
  img.putdata(list)
  img.save(path)

def make_dataset():
  x_list, y_list = [], []
  f = open('../data/cifar/data_batch_1', 'rb')
  rt = cPickle.load(f)
  x_list.extend(rt.values()[0])
  y_list.extend(rt.values()[1])
  f.close()
  train_set = tuple([x_list, y_list])
  valid_set = tuple([x_list, y_list])

  x_list, y_list = [], []
  f = open('../data/cifar/test_batch', 'rb')
  rt = cPickle.load(f)
  x_list.extend(rt.values()[0])
  y_list.extend(rt.values()[1])
  f.close()
  test_set = tuple([x_list, y_list])
  # saving
  f = file('cifar.pkl', 'wb')
  cPickle.dump([train_set, test_set, valid_set], f, -1)
  f.close()
  from subprocess import check_call
  check_call('gzip -f ' + 'cifar.pkl',shell=True)  

if __name__ == '__main__':
  datasets = load_data('cifar.pkl.gz')
  arr = datasets[0][0].get_value()[0]
  path = 'tmp.jpg'
  gen_img(arr, path)

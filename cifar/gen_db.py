import gzip
import cPickle
import pylab
import numpy

if __name__ == '__main__':
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

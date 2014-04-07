import cPickle
import gzip
from PIL import Image
from my_utils import gen_img

if __name__ == '__main__':
  dataset = 'data/caltech.pkl.gz'
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()

  l = len(train_set[0])
  for num in xrange(l):
    gen_img(test_set[0][num], 'caltech/' + str(num) + '.jpg')
  
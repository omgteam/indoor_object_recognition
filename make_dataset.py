import gzip
import cPickle
import pylab
import numpy
import theano.tensor as T
from theano import *
from PIL import Image
from theano.tensor.nnet import conv
if __name__ == '__main__':
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
	conv_out = conv.conv2d(input, W)
	output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
	f = theano.function([input], output)


	train_set_x = T.dmatrix()
	x_list, y_list = [], []
	for i in xrange(50):
		img_str = 'image_' + (4 - len(str(i + 1))) * '0' + str(i + 1) + '.jpg'
		img = Image.open(open('images/cup/' + img_str))
		img = numpy.asarray(img, dtype='float64') / 256.
		img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 150, 150)
		filtered_img = f(img_)
		filtered_img = filtered_img[0, 1, :, :]
		filtered_img = filtered_img.reshape(142 * 142)
		x_list.append(filtered_img)
		y_list.append(0)
	for i in xrange(50):
		img_str = 'image_' + (4 - len(str(i + 1))) * '0' + str(i + 1) + '.jpg'
		img = Image.open(open('images/camera/' + img_str))
		img = numpy.asarray(img, dtype='float64') / 256.
		img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 150, 150)
		filtered_img = f(img_)
		filtered_img = filtered_img[0, 1, :, :]
		filtered_img = filtered_img.reshape(142 * 142)
		x_list.append(filtered_img)
		y_list.append(1)
	for i in xrange(50):
		img_str = 'image_' + (4 - len(str(i + 1))) * '0' + str(i + 1) + '.jpg'
		img = Image.open(open('images/chair/' + img_str))
		img = numpy.asarray(img, dtype='float64') / 256.
		img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 150, 150)
		filtered_img = f(img_)
		filtered_img = filtered_img[0, 1, :, :]
		filtered_img = filtered_img.reshape(142 * 142)
		x_list.append(filtered_img)
		y_list.append(2)
	for i in xrange(50):
		img_str = 'image_' + (4 - len(str(i + 1))) * '0' + str(i + 1) + '.jpg'
		img = Image.open(open('images/cellphone/' + img_str))
		img = numpy.asarray(img, dtype='float64') / 256.
		img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 150, 150)
		filtered_img = f(img_)
		filtered_img = filtered_img[0, 1, :, :]
		filtered_img = filtered_img.reshape(142 * 142)
		x_list.append(filtered_img)
		y_list.append(3)
	train_set = tuple([x_list, y_list])
	test_set = tuple([x_list, y_list])
	valid_set = tuple([x_list, y_list])
	train_set = tuple([x_list, y_list])
	test_set = tuple([x_list, y_list])
	valid_set = tuple([x_list, y_list])

	# saving
	f = file('my_data.pkl', 'wb')
	cPickle.dump([train_set, test_set, valid_set], f, -1)
	f.close()
	from subprocess import check_call
	check_call('gzip -f my_data.pkl',shell=True)  

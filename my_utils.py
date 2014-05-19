import gzip
import cPickle
#import pylab
import numpy
import theano.tensor as T
import math
import os
from os import system
from theano import *
from theano.tensor.nnet import conv
from PIL import Image

def generate_classes(path):
	"""
	Iterate through images dir and obtain classes name list

	:type path: string
	:param path: dir of classes of images, path/Yj/Xi.jpg
	"""
	system('ls '+ path +' > tmp')
	fin=open('tmp','r')
	classes=[]
	for line in fin.readlines():
		splits=line.strip().split('.')
		classes.append(splits[len(splits)-1])
	fin.close()
	system('rm -rf tmp')
	return classes
def file_2_array(path, width, height):
	"""
	Transform image file to array
	
	:type path: string
	:param path: image path

	:type width: int
	:param width: the width of output image

	:type height: int
	:param height: the height of output image

	:type return: numpy.array(1,3,width,height)
	:param return: array of image
	"""
	if(os.path.isdir(path)):
		return None
	img = Image.open(open(path))
#  print path
#  print img.size
	img = img.resize((width,height),Image.ANTIALIAS)
#  print img.size
	img = numpy.asarray(img, dtype='float64') / 256.
#  print img.shape
	if(img.shape != (width,height,3)):
		return img
	img = img.swapaxes(0,2).swapaxes(1,2).reshape(1, 3, width, height)
	return img
def image_2_array(path, width, height):
	"""
	Transform image file to array
	
	:type path: string
	:param path: image path

	:type width: int
	:param width: the width of output image

	:type height: int
	:param height: the height of output image

	:type return: numpy.array(3*width*height)
	:param return: array of image
	"""
	if(os.path.isdir(path)):
		return None
	img = Image.open(open(path))
#  print path
#  print img.size
	img = img.resize((width,height),Image.ANTIALIAS)
#  print img.size
	img = numpy.asarray(img, dtype='float64') / 256.
#  print img.shape
	if(img.shape != (width,height,3)):
		return img
	img = img.swapaxes(0,2).swapaxes(1,2).reshape(3*width*height)
	return img

def create_formatted_pkl(ori_images_dir, pkl_images_path, train_ratio, ishape):
	"""
	Create pkl file using multi-class images

	:type ori_images_dir: string
	:param ori_images_dir: dir/Yj/Xi.jpg

	:type pkl_images_path: string
	:param pkl_images_path: output pkl file's path

	:type train_ratio: float
	:param train_ratio: percentage of what should be training set

	:type ishape: tuple 
	:param ishape: (width,height)
	"""
	train_x_list,train_y_list=[],[]
	valid_x_list,valid_y_list=[],[]

	system('ls '+ ori_images_dir+' > tmp')
	fin=open('tmp','r')
	fout=open('shape_fail_images','w')
	class_index=0
	for line in fin.readlines():
		line=line.strip()
		class_dir=ori_images_dir+'/'+line
		system('ls '+class_dir+' > tmp_class')
		images_path=[]
		fin2=open('tmp_class','r')
		for line_image in fin2.readlines():
			images_path.append(class_dir+'/'+line_image.strip())
		num_train=int(train_ratio*len(images_path))
		for i in xrange(num_train):
			img=image_2_array((images_path[i]),ishape[0],ishape[1])
			if(img != None and len(img)==3*ishape[0]*ishape[1]):
				train_x_list.append(img)
				train_y_list.append(class_index)
			else:
				fout.write(images_path[i])
		for i in xrange(num_train, len(images_path)):
			img=image_2_array((images_path[i]),ishape[0],ishape[1])
			if(img != None and len(img)==3*ishape[0]*ishape[1]):
				valid_x_list.append(img)
				valid_y_list.append(class_index)
			else:
				fout.write(images_path[i])
		class_index=class_index+1
		fin2.close()
	fin.close()
	fout.close()
	system('rm -rf tmp')
	system('rm -rf tmp_class')
	print len(train_x_list),len(train_y_list)
	print len(valid_x_list),len(valid_y_list)
	train_set=tuple([train_x_list, train_y_list])
	test_set=tuple([valid_x_list, valid_y_list])
	valid_set=tuple([valid_x_list, valid_y_list])

	# saving
	f = file(pkl_images_path, 'wb')
	cPickle.dump([train_set, test_set, valid_set], f, -1)
	f.close()
	from subprocess import check_call
	check_call('gzip -f ' + pkl_images_path,shell=True)  

def shuffle_list(x_list,y_list):
	"""
	Shuffle x_list,y_list randomly, without changing coressponding yi for each xi

	:type x_list: list
	:param x_list: images' arrays list

	:type y_list: list
	:param y_list: images' classes list
	"""
	x_len=len(x_list)
	y_len=len(y_list)
	print x_len
	assert x_len==y_len
	import random
	for i in xrange(x_len):
		swap_index=random.randint(i,x_len-1)
		tmp=x_list[i]
		x_list[i]=x_list[swap_index]
		x_list[swap_index]=tmp
		tmp=y_list[i]
		y_list[i]=y_list[swap_index]
		y_list[swap_index]=tmp

def subtract_list(x_list):
	for i in xrange(len(x_list)):
		x_list[i]=x_list[i]-numpy.mean(x_list[i])

def nor_list(x_list):
	for i in xrange(len(x_list)):
		x_list[i]=x_list[i]/numpy.std(x_list[i])
def abs_list(x_list):
	for i in xrange(len(x_list)):
		x_list[i]=numpy.abs(x_list[i])

def load_data(dataset):
	"""
	Load pkl data into memory, namely[(xi,yi)]
	
	:type dataset: string
	:param dataset: pkl path. pkl is created by func my_utils.create_formatted_pkl
	"""
	f = gzip.open(dataset+'.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)

	train_x_list=train_set[0]
	train_y_list=train_set[1]
	test_x_list=test_set[0]
	test_y_list=test_set[1]
	valid_x_list=valid_set[0]
	valid_y_list=valid_set[1]
	from parameters import sub_mode,nor_mode,abs_mode
	if(sub_mode==1):
		subtract_list(train_x_list)
		subtract_list(test_x_list)
		subtract_list(valid_x_list)
	if(nor_mode==1):
		nor_list(train_x_list)
		nor_list(test_x_list)
		nor_list(valid_x_list)
	if(abs_mode==1):
		abs_list(train_x_list)
		abs_list(test_x_list)
		abs_list(valid_x_list)
	shuffle_list(train_x_list,train_y_list)
	shuffle_list(test_x_list,test_y_list)
	shuffle_list(valid_x_list,valid_y_list)
	
	train_set=tuple([train_x_list,train_y_list])
	valid_set=tuple([valid_x_list,valid_y_list])
	test_set=tuple([test_x_list,test_y_list])
	f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

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

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
		(test_set_x, test_set_y)]
	return rval

def download_urls(input_path,output_dir):
	"""
	Download image given specified urls
	"""
	fin=open(input_path,'r')
	for line in fin.readlines():
		input_path=input_path.strip()
		splits=input_path.split('/')
		img_name=splits[len(splits)-1]
		os.system('wget '+line+' -o '+output_dir+'/'+img_name)
def generate_probability_model(relation_pairs_path, class_int_dict):
	"""
	Generate probabiliry model to improve classification accuracy.

	:type relation_pairs_path: file path of class_a,class_b pairs
	"""
	from parameters import num_classes
	fin=open(relation_pairs_path)
	fre_matrix=numpy.array(numpy.ones(num_classes**2,dtype='int32').reshape((num_classes,num_classes)))
	sum_vector=[num_classes*1.0 for i in xrange(num_classes)]
	for line in fin.readlines():
		line=line.strip()
		class_i,class_j=line.split()
		i=class_int_dict[class_i]
		j=class_int_dict[class_j]
		fre_matrix[i][j]=fre_matrix[i][j]+1
		fre_matrix[j][i]=fre_matrix[j][i]+1
		sum_vector[i]=sum_vector[i]+1
		sum_vector[j]=sum_vector[j]+1
	pro_matrix=numpy.array(numpy.zeros(num_classes**2,dtype='float32').reshape((num_classes,num_classes)))
	for i in xrange(num_classes):
		for j in xrange(num_classes):
			pro_matrix[i][j]=fre_matrix[i][j]/sum_vector[j]
	return pro_matrix


if __name__ == '__main__':
	from parameters import indoor_images_urls_path,indoor_images_dir
	download_urls(indoor_images_urls_path,indoor_images_dir)

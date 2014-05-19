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
from parameters import *

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
def get_classify_func():
	"""
	Reconstruct classify model using training output

	:type return: theano.function
	:param return: function that return class index given image array(numpy.array(1,3,ishape[0],ishape[1])
	"""
	n_in = ishape[0]*ishape[1]
	n_out = log_n_out

	f = open(params_file)
	best_params = cPickle.load(f)
	f.close()

	rng = numpy.random.RandomState(23455)
	# classify
	x = T.dmatrix('x')
	y = T.lvector('y')
	
	conv_pool_input = x.reshape(classify_image_shapes[0])
	conv_pool_layers=[]
	conv_pool_layers.append(LeNetConvPoolLayer(rng, input=conv_pool_input,
		image_shape=classify_image_shapes[0],
		filter_shape=filter_shapes[0],
		poolsize=pool_sizes[0]))
	conv_pool_layers[0].W.set_value(best_params[0].get_value(), borrow=True)
	conv_pool_layers[0].b.set_value(best_params[1].get_value(), borrow=True)
	i=1;
	param_index=2
	while i < num_conv_pool_layers:
		conv_pool_layers.append(LeNetConvPoolLayer(rng,
	  		 	input=conv_pool_layers[i-1].output,
	    		image_shape=classify_image_shapes[i],
		    	filter_shape=filter_shapes[i], poolsize=pool_sizes[i]))
		conv_pool_layers[i].W.set_value(best_params[param_index].get_value(), borrow=True)
		param_index=param_index+1
		conv_pool_layers[i].b.set_value(best_params[param_index].get_value(), borrow=True)
		param_index=param_index+1
		i=i+1

	mlp_layers=[]
	mlp_input = conv_pool_layers[num_conv_pool_layers-1].output.flatten(2)
	mlp_layers.append(HiddenLayer(rng, input=mlp_input, n_in=mlp_n_in[0],
				n_out=mlp_n_out[0], activation=T.tanh))
	mlp_layers[0].W.set_value(best_params[param_index].get_value(), borrow=True)
	param_index=param_index+1
	mlp_layers[0].b.set_value(best_params[param_index].get_value(), borrow=True)
	param_index=param_index+1
	i=1
	while i<num_mlp_layers:
		mlp_layers.append(HiddenLayer(rng, input=mlp_layers[i-1].output, n_in=mlp_n_in[i],
			n_out=mlp_n_out[i], activation=T.tanh))
		mlp_layers[i].W.set_value(best_params[param_index].get_value(), borrow=True)
		param_index=param_index+1
		mlp_layers[i].b.set_value(best_params[param_index].get_value(), borrow=True)
		param_index=param_index+1
		i=i+1

	# classify the values of the fully-connected sigmoidal layer
	log_layer= LogisticRegression(input=mlp_layers[num_mlp_layers-1].output, n_in=mlp_n_out[num_mlp_layers-1], n_out=log_n_out)

	log_layer.W.set_value(best_params[param_index].get_value(), borrow=True)
	param_index=param_index+1
	log_layer.b.set_value(best_params[param_index].get_value(), borrow=True)
	param_index=param_index+1
	# compiled theano function that returns this value
	classify = theano.function(inputs=[conv_pool_input], outputs=(log_layer.y_pred,log_layer.p_y_given_x))
	return classify

def classify_image(path=None,image=None,classify_func=None):
	"""
	Classify an image to class index

	:type path: string
	:param path: image path

	:type image: numpy.array(1,3,ishape[0],ishape[1])
	:param image: image to be classified

	:type classify_func: theano.function
	:param classify_func: classification function using training output
	"""
	if(path==None and image==None):
		return None
	if(classify_func == None):
		classify_func=get_classify_func()
	if(image==None):
		image=file_2_array(path, ishape[0], ishape[1])
	res = classify_func(image)
	return res
if __name__ == '__main__':
	classify_image('test.jpg')

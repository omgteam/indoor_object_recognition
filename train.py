import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from parameters import *
if __name__ == '__main__':

	rng = numpy.random.RandomState(23455)
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size

    # allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of

    ######################
    # BUILD ACTUAL MODEL #
    ######################
	print '... building the model'
	conv_pool_input = x.reshape((batch_size, 3, ishape[0], ishape[1]))
	conv_pool_layers=[]
    # filtering: 80-5+1=76
    # 76/4=19
	conv_pool_layers.append(LeNetConvPoolLayer(rng, input=conv_pool_input,
		image_shape=image_shapes[0],
		filter_shape=filter_shapes[0],
		poolsize=pool_sizes[0]))
	i=1;
	while i < num_conv_pool_layers:
		conv_pool_layers.append(LeNetConvPoolLayer(rng,
	  	 	input=conv_pool_layers[i-1].output,
	    	image_shape=image_shapes[i],
		    filter_shape=filter_shapes[i], poolsize=pool_sizes[i]))
		i=i+1

	mlp_layers=[]
	mlp_input = conv_pool_layers[num_conv_pool_layers-1].output.flatten(2)
	mlp_layers.append(HiddenLayer(rng, input=mlp_input, n_in=mlp_n_in[0],
                         n_out=mlp_n_out[0], activation=T.tanh))
	i=1
	while i<num_mlp_layers:
		mlp_layers.append(HiddenLayer(rng, input=mlp_layers[i-1].output, n_in=mlp_n_in[i],
						n_out=mlp_n_out[i], activation=T.tanh))
		i=i+1
	
    # classify the values of the fully-connected sigmoidal layer
	log_layer= LogisticRegression(input=mlp_layers[num_mlp_layers-1].output, n_in=mlp_n_out[num_mlp_layers-1], n_out=log_n_out)

    # the cost we minimize during training is the NLL of the model
	cost = log_layer.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
	test_model = theano.function([index], log_layer.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

	validate_model = theano.function([index], log_layer.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
	params=[]
	i=0
	while i<num_conv_pool_layers:
		params+=conv_pool_layers[i].params
		i=i+1
	i=0
	while i<num_mlp_layers:
		params+=mlp_layers[i].params
		i=i+1

	params += log_layer.params

    # create a list of gradients for all model parameters
	grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
	updates = []
	for param_i, grad_i in zip(params, grads):
		updates.append((param_i, param_i - learning_rate * grad_i))

	train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
	print '... training'
	validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()
	epoch = 0
	done_looping = False
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index
			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					best_params=[]
					i=0
					while i<num_conv_pool_layers:
						best_params+=conv_pool_layers[i].params
						i=i+1
					i=0
					while i<num_mlp_layers:
						best_params+=mlp_layers[i].params
						i=i+1

					best_params += log_layer.params
                    #improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
						patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

                    # test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

	print '... saving params'
	f = file(params_file, 'wb')
	cPickle.dump(best_params, f, -1)
	f.close()

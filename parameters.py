learning_rate=0.04
n_epochs=10
batch_size=10
dataset='../caltech/caltech_80.pkl.gz'
num_classes=13
ishape = (80, 80)

#conv_pool layer info
num_conv_pool_layers=3;
nkerns=[16,96,2400]
pool_sizes=[(4,4),
            (2,2),
            (1,1)]
filter_shapes=[(nkerns[0], 3, 5, 5),
               (nkerns[1], nkerns[0], 6, 6),
               (nkerns[2], nkerns[1], 6, 6)]
image_shapes=[(batch_size,3, ishape[0], ishape[1])]

i=1
while i<=num_conv_pool_layers:
	image_shapes.append((batch_size,
						 nkerns[i-1],
						(image_shapes[i-1][2]-filter_shapes[i-1][2]+1)/pool_sizes[i-1][0],
						(image_shapes[i-1][3]-filter_shapes[i-1][3]+1)/pool_sizes[i-1][1]))
	i=i+1

i=num_conv_pool_layers-1
conv_pool_out = nkerns[i]*(image_shapes[i][2]-filter_shapes[i][2]+1)/pool_sizes[i][0]*(image_shapes[i][3]-filter_shapes[i][3]+1)/pool_sizes[i][1]
#mlp layers info
num_mlp_layers=2
mlp_n_in=[conv_pool_out,500]
mlp_n_out=[500,500]

#log layer info
log_n_in=mlp_n_out[num_mlp_layers-1]
log_n_out=num_classes

# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
improvement_threshold = 0.995  # a relative improvement of this much is

                                   # considered significant
params_file='lenet_params.save'


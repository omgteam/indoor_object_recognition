from my_utils import generate_classes
#datasets
ori_images_dir='/root/james/dataset/indoor_object_images'
pkl_images_path='/root/james/dataset/indoor_object_images.pkl'
#ori_images_dir='/root/james/dataset/man_clean'
#pkl_images_path='/root/james/dataset/man_clean.pkl'
ishape = (80, 80)
image_classes=generate_classes(ori_images_dir)
class_int_dict={}
for i in xrange(len(image_classes)):
	class_int_dict[image_classes[i]]=i
	
train_ratio = 0.9

learning_rate=0.04
n_epochs=100
batch_size=50
dataset=pkl_images_path
#dataset='/root/james/caltech/caltech_80.pkl'
num_classes=len(image_classes)

#conv_pool layer info
num_conv_pool_layers=3;
#nkerns=[16,96,2400]
nkerns=[16,64,256]
pool_sizes=[(4,4),
            (2,2),
            (2,2)]
filter_shapes=[(nkerns[0], 3, 5, 5),
               (nkerns[1], nkerns[0], 6, 6),
               (nkerns[2], nkerns[1], 6, 6)]
image_shapes=[(batch_size,3, ishape[0], ishape[1])]
classify_image_shapes=[(1,3, ishape[0], ishape[1])]

i=1
while i<=num_conv_pool_layers:
	image_shapes.append((batch_size,
		nkerns[i-1],
		(image_shapes[i-1][2]-filter_shapes[i-1][2]+1)/pool_sizes[i-1][0],
		(image_shapes[i-1][3]-filter_shapes[i-1][3]+1)/pool_sizes[i-1][1]))
	classify_image_shapes.append((1,
		nkerns[i-1],
		(image_shapes[i-1][2]-filter_shapes[i-1][2]+1)/pool_sizes[i-1][0],
		(image_shapes[i-1][3]-filter_shapes[i-1][3]+1)/pool_sizes[i-1][1]))
	i=i+1

i=num_conv_pool_layers-1
conv_pool_out = nkerns[i]*(image_shapes[i][2]-filter_shapes[i][2]+1)/pool_sizes[i][0]\
						 *(image_shapes[i][3]-filter_shapes[i][3]+1)/pool_sizes[i][1]
#mlp layers info
num_mlp_layers=1
mlp_n_in=[conv_pool_out,64]
mlp_n_out=[64,32]

#log layer info
log_n_in=mlp_n_out[num_mlp_layers-1]
log_n_out=num_classes

# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
improvement_threshold = 0.995  # a relative improvement of this much is

                                   # considered significant
params_file='lenet_params.save'

indoor_images_urls_path='/root/james/dataset/indoor_images_urls'
indoor_images_dir='/root/james/dataset/indoor_images'

relation_pairs_path='/root/james/dataset/indoor_objects_pairs'
threshold_p_y_given_x=0.0
model_factor=0.5

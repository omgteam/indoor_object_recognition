#############
#path config#
#############

ori_images_dir='/root/james/dataset/indoor_object_images'
'''$(ori_images_dir)/Yi_dir/Xj_img, here we use caltech256'''

pkl_images_path='/root/james/dataset/indoor_object_images.pkl'
'''Image arrary package, [[Xi,Yj]]. If not found, will transfer images in $(ori_images_dir) to pkl, the method is in my_utils.py---create_formatted_pkl'''

params_file='lenet_params.save'
'''The file path to save optimal parameters of model obtained through training'''

indoor_images_urls_path='/root/james/dataset/indoor_images_urls'
'''Indoor images urls path'''

indoor_images_dir='/root/james/dataset/indoor_images'
'''Download image urls specified in $(indoor_images_urls_path)'''

relation_pairs_path='/root/james/dataset/indoor_objects_pairs'
'''File path to store adjacent object pairs that apear in indoor objects, [(Yi,Yj)]'''

ishape = (80, 80)
'''Target image shape. original images may be various shape, and will be transform into ishape'''

from my_utils import generate_classes
image_classes=generate_classes(ori_images_dir)
'''Get classes(Yi) in $(ori_images_dir)'''

class_int_dict={}
'''Map class name to int'''
for i in xrange(len(image_classes)):
	class_int_dict[image_classes[i]]=i
	
train_ratio = 0.9
'''Divide {(xi,yi)} into training set and valid set, 0.9 means 90% of data will be included in trainning set, and 10% valid set'''

learning_rate=0.04
''' Model learning fator'''

n_epochs=100
'''Number of iteration through dataset during training process'''

batch_size=50
'''Number of (xi,yi) pairs to train at a time during trainging process'''

dataset=pkl_images_path
num_classes=len(image_classes)

#conv_pool layer info
num_conv_pool_layers=2;
'''Number of CNN layers in learning model. Each layer consist of conv, pool process'''

#nkerns=[16,96,2400]
nkerns=[64,256]
'''Number of feature maps. Here, first layer has 64 maps, and second one has 256 maps'''

pool_sizes=[(4,4),
            (4,4)]
'''Window size of max sampling pool'''

filter_shapes=[(nkerns[0], 3, 5, 5),
               (nkerns[1], nkerns[0], 4, 4)]
'''[(number of output feature maps, number of input feature maps,Window size of convolution)]'''

image_shapes=[(batch_size,3, ishape[0], ishape[1])]
classify_image_shapes=[(1,3, ishape[0], ishape[1])]
'''Batch_size of image_shapes denotes number of images input at a time. 3 denotes RGB. the first 1 in classify_image_shapes implies that one image can be classify at a time'''

#caculate image_shapes by previous image shape, filter shape and pool size
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

num_mlp_layers=0
#1,128
mlp_n_in=[conv_pool_out,64]
mlp_n_out=[64,conv_pool_out]

#log layer info
if(num_mlp_layers!=0):
	log_n_in=mlp_n_out[num_mlp_layers-1]
else:
	log_n_in=conv_pool_out
log_n_out=num_classes

# early-stopping parameters
patience = 100000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
improvement_threshold = 0.995  # a relative improvement of this much is

                                   # considered significant
threshold_p_y_given_x=0.05
model_factor=0.5

#sub_mode=1 image[i][j]=image[i][j]-mean(image[i])
#nor_mode=1 image[i][j]=image[i][j]/standar_deviation(image[i])
#abs_mode=1 image[i][j]=abs[image[i][j])
sub_mode=0
nor_mode=0
abs_mode=0

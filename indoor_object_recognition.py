from PIL import Image
from scipy import ndimage
import numpy
from skimage import filter
from skimage import morphology
from parameters import *
from my_utils import generate_probability_model
from classify import get_classify_func,classify_image
def get_object_images(image_path,output_shape):
	img=Image.open(image_path)
	grey_img=numpy.asarray(img.convert('L'))
	otsu_val=filter.threshold_otsu(grey_img)
	mask=grey_img<otsu_val
	filtered_img=ndimage.gaussian_filter(mask,sigma=1/(4.*output_shape[0]))
	blobs=filtered_img>filtered_img.mean()
	blobs_labels=morphology.label(blobs,background=0)
	
	label_im,nb_labels=ndimage.label(blobs_labels)
	objects_slices=ndimage.find_objects(label_im)
	boxes=[]
	num_imgs=0
	for i in xrange(nb_labels):
		x,y=objects_slices[i]
		box=(x.start,y.start,x.stop,y.stop)
		if(box[2]-box[0]<output_shape[0]/2 or box[3]-box[1] <output_shape[1]/2):
			continue
		num_imgs=num_imgs+1
		boxes.append(box)
	object_images=[]
	for i in xrange(num_imgs):
		object_images.append(img.crop(boxes[i]).resize(output_shape,Image.ANTIALIAS))
#		object_images.append(img.crop(boxes[i]))
	return object_images

if __name__=='__main__':
	print 'generating deep learning model...'
	classify_func=get_classify_func()
	print 'generating probabilty model...'
	pro_model = generate_probability_model(relation_pairs_path,class_int_dict)
	
	print 'combining these two models'
	test_path='test.jpg'
	test_name=test_path.split('.')[0]
	images=get_object_images(test_path,ishape)
	last_object=None
	for i in xrange(len(images)):
		img = numpy.asarray(images[i], dtype='float64') / 256.
		#  print img.shape
		if(img.shape != (ishape[0],ishape[1],3)):
			continue
		img = img.swapaxes(0,2).swapaxes(1,2).reshape(1,3,ishape[0],ishape[1])
		y_pred,p_y_given_x=classify_image(image=img,classify_func=classify_func)
		y_pred=y_pred[0]
		p_y_given_x=p_y_given_x[0]
		if(p_y_given_x[y_pred]<threshold_p_y_given_x):
			continue
		print 'deep learning model implies this is:\n',image_classes[y_pred]
		if(last_object==None):
			images[i].save(test_name+'_output_object_'+str(i)+'_'+image_classes[y_pred]+'.jpg')
			last_object=y_pred
		else:
			final_p_y_given_x=p_y_given_x + model_factor * pro_model[last_object]
			p_dict={final_p_y_given_x[i]:i for i in xrange(len(final_p_y_given_x))}
			max_value=max(final_p_y_given_x)
			if(max_value<threshold_p_y_given_x):
				continue
			final_y_pred=p_dict[max_value]	
			print 'final model judge that this is :\n',image_classes[final_y_pred]
			images[i].save(test_name+'_output_object_'+str(i)+'_'+image_classes[final_y_pred]+'.jpg')
			last_object = final_y_pred

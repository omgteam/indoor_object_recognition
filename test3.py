from indoor_object_recognition import *
img_path='test6.jpg'
imgs=get_object_images(img_path,(80,80))
num=len(imgs)
print num
for i in xrange(num):
	imgs[i].save('crop_'+img_path+'_'+str(i)+'.jpg')

from PIL import Image
import numpy
img_path='test.jpg'
img=Image.open(open(img_path))
img2=Image.open(img_path)
img=numpy.asarray(img,dtype='float64')/256.
img2=img2.resize((40,40),Image.ANTIALIAS)
img2=numpy.asarray(img2,dtype='float64')/256.
print img.shape
print img[0][0:10][0:10]
print img2.shape
print img2[0][0:10][0:10]

import cv2
import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from skimage import color
import random
from matplotlib import pyplot as plt
from PIL import Image 
from scipy.misc import imsave
import numpy



def binarize_array(numpy_array, threshold=150):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array
    
    
    
def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    imsave(target_path, image)

var = os.listdir("img")
dir1="./img/"
#for i in var:
imgpath = os.path.join(dir1,var[0])
"""#img1 = cv2.imread(imgpath, 0)
	image_file = Image.open(imgpath) # open colour image
	#img1 = color.rgb2gray(img1)
	image_file = image_file.convert('L') # convert image to black and white
	image_file.save('result.jpg')
	#plt.imshow(img1)
	#plt.show()
	img1 = cv2.imread(imgpath, 0)
	img1 = color.rgb2gray(img1)
	plt.imshow(img1)
	plt.show()
	#print(img1.shape)"""
imgpath1 = "./result.jpg"
img1 = cv2.imread(imgpath1, 0)
img = cv2.resize(img1,(1024,1024))
cv2.imwrite('d.jpg',img)
imgpath2 = "./d.jpg"
binarize_image(imgpath2, "e.jpg", 100)
	





"""
def get_parser():
    #Get parser object for script xy.py.
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",
                        dest="input",
                        help="read this file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="write binarized file hre",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--threshold",
                        dest="threshold",
                        default=200,
                        type=int,
                        help="Threshold when to show white")
    return parser
"""

#if __name__ == "__main__":
    #args = get_parser().parse_args()
   # binarize_image(args.input, args.output, args.threshold)
   
""" i think we can expect that user can crop the lines for us.
then all what we have to id slide the box and then detect the letters and symbols.
if there are two or more lines , it will definitely crop one after the another in a sequence.
if there is just one line, maybe we dont need it crop.
we need to find a way to detect it in an image
"""


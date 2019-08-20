import cv2
import numpy as np
from pytesseract import image_to_string
import pytesseract


def pre_process(img):
    #blurr image to smooth 
    blurr = cv2.GaussianBlur(img, (3,3),0)

    #convert image to grayscale
    grey = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    grey = cv2.filter2D(grey, -1, kernel)
    
    _,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY_INV)
    
    return thresh


def find_table(thresh, img):

        # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80
        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1)) 
        # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(thresh, verticalStructure, iterations=3)
    vertical = cv2.dilate(img_temp1, verticalStructure, iterations=3)
    
    img_temp2 = cv2.erode(thresh, horizontalStructure, iterations=3)
    horizontal = cv2.dilate(img_temp2, horizontalStructure, iterations=3)
 # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha   
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical, alpha, horizontal, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    table=img_final_bin

    return table


def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def img_rescaling(new_img):
        row,col = new_img.shape[:2]  
        row=row*1.2
        col=col*1.2
        new_img=cv2.resize(new_img,(int(col),int(row)))
        return new_img


def img_to_Text(img):
    output = pytesseract.image_to_string(img)
    return output



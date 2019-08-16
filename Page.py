import cv2
import numpy as np
from pytesseract import image_to_string
import pytesseract
import imutils
from PIL import Image
import PIL.Image
import os

def pre_process(img):
    #blurr image to smooth 
    blurr = cv2.GaussianBlur(img, (3,3),0)

    #convert image to grayscale
    grey = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    grey = cv2.filter2D(grey, -1, kernel)
    
    _,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY_INV)
        
    # MAX_THRESHOLD_VALUE = 255
    # BLOCK_SIZE = 15
    # THRESHOLD_CONSTANT = 0

    
    # # Filter image
    # filtered = cv2.adaptiveThreshold(grey, MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)

    
    return thresh

def find_table(thresh, isTable):
    #Extract cols and rows of image
    height, width = thresh.shape[:2]
    scale = 35 #play with this variable in order to increase/decrease the amount of lines to be detected
    
    #Specify size on horizontal axis
    horizontalsize =  int(width / scale)
    verticalsize = int(height / scale)
    
    #Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
    
    #Apply morphology operations
    horizontal = cv2.erode(thresh,horizontalStructure)
    horizontal = cv2.dilate(horizontal,horizontalStructure)
    
    vertical = cv2.erode(thresh,verticalStructure)
    vertical = cv2.dilate(vertical,verticalStructure)
    
    if isTable==1:
        return vertical
    
    table=horizontal+vertical

    return table
    
def img_to_Text(path,QA_flag):
    output = 'Start'+'\n'+'----------------------------'+'\n'

    n = cv2.imread(path)
    output = pytesseract.image_to_string(n,lang='Eng')
    if QA_flag==0:
            np.savetxt('Questions.txt',[output], fmt='%s')
    elif QA_flag==1:
            np.savetxt('Answers.txt',[output], fmt='%s')
    else:
            np.savetxt('Table.txt',[output], fmt='%s')
    dir_name = "Tables/"
    test = os.listdir(dir_name)   
    for item in test:
            if item.endswith(".png"):
                    os.remove(os.path.join(dir_name, item))



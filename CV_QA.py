import cv2
import numpy as np
import Page
from pytesseract import image_to_string
import pytesseract
import imutils
from PIL import Image
import PIL.Image
import os

q_path="Tables/Questions/"
a_path="Tables/Answers/"


Page.emptyDIR(q_path)
Page.emptyDIR(a_path)

img = cv2.imread("input_images/1w.jpeg",1)
thresh = Page.pre_process(img)
table = Page.find_table(thresh, img)
# Find contours for image, which will detect all the boxes
im2, contours, hierarchy = cv2.findContours(table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = Page.sort_contours(contours, method="top-to-bottom")
idx = 0
Qs=""
As=""
for c in contours:
        area=float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        if (w > 80 and h > 10) and w > 3*h and area<20000:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            if x<150:
                #    new_img=Page.pre_process(new_img)
                   Qs+='->'+Page.img_to_Text(new_img)+'\n'
                    
            else:
                #    new_img=Page.pre_process(new_img)
                   As+='->'+Page.img_to_Text(new_img)+'\n'
            cv2.imshow("Box",new_img)
            cv2.waitKey(0)
            
print("Questions")
print(Qs)
print("Answers")
print(As)
cv2.waitKey(0)
cv2.destroyAllWindows()


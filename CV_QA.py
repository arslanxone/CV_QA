import cv2
import numpy as np
import Page
from pytesseract import image_to_string
import pytesseract

show_steps=False #Please change this to True if you want to see the steps

img = cv2.imread("input_images/3.jpeg",1)
thresh = Page.pre_process(img)
table = Page.find_table(thresh, img)
# Find contours for image, which will detect all the boxes
im2, contours, hierarchy = cv2.findContours(table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = Page.sort_contours(contours, method="top-to-bottom")
idx = 0
Qs=""
As=""
Box_img = img.copy()
for c in contours:
        area=float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        if (w > 80 and h > 10) and w > 3*h and area<20000:
            idx += 1
            cv2.rectangle(Box_img,(x,y),(x+w,y+h),(0,0,255),1)
            new_img = img[y:y+h, x:x+w]
            if x<150:
                #    If you want to increase result of OCR please uncomment this code. 
                #     we can add more filters according to the noise in our image 
                #      new_img=Page.pre_process(new_img)   #Converted into thresholded image
                #      new_img=Page.img_rescaling(new_img) #Resizing image
                     Qs+='->'+Page.img_to_Text(new_img)+'\n'
                    
            else:
                #    If you want to increase result of OCR please uncomment this code 
                #     we can add more filters according to the noise in our image
                #      new_img=Page.pre_process(new_img) #Converted into thresholded image
                #      new_img=Page.img_rescaling(new_img) #Resizing image
                     As+='->'+Page.img_to_Text(new_img)+'\n'

print("Questions")
print(Qs)
print("Answers")
print(As)

if show_steps==True:

        cv2.imshow("Step1_Input_Image",img)
        cv2.imshow("Step2_Preprocessed_Image",thresh)
        cv2.imshow("Step3_Tables",table)
        cv2.imshow("Step4_Detected_Box",Box_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()


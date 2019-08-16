import cv2
import numpy as np
import Page
from pytesseract import image_to_string
import pytesseract
import imutils
from PIL import Image
import PIL.Image
import os

#Page Size
PAGE_W=480
PAGE_H=700

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
        


img = cv2.imread("input_images/1.jpeg",1)
thresh = Page.pre_process(img)
table = Page.find_table(thresh, 0)
# cv2.namedWindow('Tables',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Tables',table)
# cv2.waitKey(0)
 # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
_, contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for i in contours:
    area=float(cv2.contourArea(i))
    elip =  cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i,0.04*elip, True)
    if len(approx)==4 and area>50000 and area<100000:             
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=img[y+1:y+h,x:x+w]
        r_y, r_x, _=roi.shape
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY_INV)
        vertical=Page.find_table(thresh, 1)
        _, v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = 0   
        for j in v_contours:
                v_area=float(cv2.contourArea(j))
                v_elip =  cv2.arcLength(j, True)
                v_approx = cv2.approxPolyDP(j,0.04*elip, True)
                # print(v_area,len(v_approx))
                if len(v_approx)==2 and v_area>80: 
                        cnt= cnt + 1
                        if cnt==2: 
                                a,b,c,d = cv2.boundingRect(j)
                                if a<r_x and a!=0:
                                        Qs=thresh[1:r_y,1:a]
                                        As=thresh[1:r_y,a:r_x]
                                        cv2.imwrite("Tables/Qs.TIFF",Qs)
                                        cv2.imwrite("Tables/As.TIFF",As)
                                        cv2.imshow('Questions',Qs)
                                        cv2.imshow('Answers',As)
                                        Page.img_to_Text("Tables/Qs.TIFF",0)
                                        Page.img_to_Text("Tables/As.TIFF",1)

                                elif a==r_x or a==0:
                                        cv2.imwrite("Tables/table.TIFF",thresh)
                                        Page.img_to_Text("Tables/table.TIFF",2)



#resize image just to view the output properly
img = cv2.resize(img,(PAGE_W,PAGE_H))
cv2.imshow('Output_Detected_Table',img)
#cv2.imshow('Table',thresh) 
# cv2.imshow('Questions',Qs)
# cv2.imshow('Answers',As)
cv2.waitKey(0)
cv2.destroyAllWindows()


import csv
import cv2, os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd

def png_to_csv(filename):
    """takes a file in the form of png and converts it into a csv with the pixel values
    """

    img = cv2.imread(filename)
    img = cv2.resize(img, (8,8) ) # resize image
    if len(img.shape) == 3:  # if three channels
        img = img[:,:,0]     # just take one
    

    flattened_array = img.flatten()
    print(f"flattened_array is {flattened_array}")
    f = open("wyatt10by10.csv", "w", newline='')
    w = csv.writer(f)
    row = flattened_array.tolist()  # convert to Python list
    row += [ 0 ]       # let's add the label
    ROW_ARRAY = [ row, row ]   # could have hundreds, we'll use two


    for r in ROW_ARRAY:  # for each row
        w.writerow(r)    # write out that row
    #print(w)
    f.close()



# def main():
#     """
#     """
#     dim = (8,8)
#     for c in range(2):
#         img = cv2.imread(f"mask{c}.png")
#         img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
#         img = np.array(img)
#         #print(img)
#         img = img.flatten()
#         print(img)
#         img.tolist()
#         #print(img)
#     print(len(img))

# main()

# img = cv2.imread('mask18.png')
# img = cv2.resize(img, (8,8) ) # resize image
# cv2.imwrite('mask8x8.png', img)

# img1 = cv2.imread('face18.png')
# img1 = cv2.resize(img1, (8,8) ) # resize image
# cv2.imwrite('face8x8.png', img1)

face_cascade = cv2.CascadeClassifier('/Users/wyattchang/Desktop/CS35_Final/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) #Capture Live Video Feed

c = 0

#F = []
while c<200:
    ret, img = cap.read()
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces =  face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces: 
        face = img[y:y+h,x:x+w]
        face = cv2.resize(face, (8,8), ) #resize to an 8x8 image
        cv2.imwrite(f"face{c}.png",face)  #cv2.savefile save to file
        #png_img = cv2.imshow(f"mask{c}.png",)

        c += 1
        #F += [1,42,18]
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
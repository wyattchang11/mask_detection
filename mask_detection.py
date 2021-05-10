# Mask Recognition
#
# Wyatt Chang
#
# Mask Detection Project


import csv
import cv2, os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import keyboard
from PIL import Image



def png_to_csv(filename):
    """takes a file in the form of png and converts it into a csv with the pixel values
    """

    img = cv2.imread(filename)
    img = cv2.resize(img, (8,8) ) # resize image
    if len(img.shape) == 3:  # if three channels
        img = img[:,:,0]     # just take one
    

    flattened_array = img.flatten()
    #print(f"flattened_array is {flattened_array}")
    #w = csv.writer(f)
    row = flattened_array.tolist()  # convert to Python list
    row += [ 0 ]       # let's add the label
    #f.close
    #print(row)
    return row
    # ROW_ARRAY = [ row, row ]   # could have hundreds, we'll use two


    # for r in ROW_ARRAY:  # for each row
    #     w.writerow(r)    # write out that row
    # print(w)
    # f.close()

def png_to_csv2(filename):
    """takes a file in the form of png and converts it into a csv with the pixel values
    """

    img = cv2.imread(filename)
    img = cv2.resize(img, (8,8) ) # resize image
    if len(img.shape) == 3:  # if three channels
        img = img[:,:,0]     # just take one
    

    flattened_array = img.flatten()
    #print(f"flattened_array is {flattened_array}")
    #f = open("wyatt10by10.csv", "w", newline='')
    #w = csv.writer(f)
    row = flattened_array.tolist()  # convert to Python list
    row += [ 1 ]       # let's add the label
    #f.close
    #print(row)
    return row

def write_to_csv( list_of_rows, filename ):
    """ writes csv 
        + input:  csv_file_name, the name of a csv file
        and a list of rows
        + output: a csv file named filename with the list of rows within it
    """
    # open file to write into
    try:
        csvfile = open( filename, "w", newline='' )
        
        filewriter = csv.writer( csvfile, delimiter=",")
        
        # write into csv
        for row in list_of_rows:
            filewriter.writerow( row )
        
        csvfile.close()

    # if no files, print error message
    except:
        print("File", filename, "could not be opened for writing...")


class NeuralNet:
    def __init__(self):
        """ Neural Network constructor
        runs the neural networks and trains it
        defines the member variables of the neural net class
        """
        print("+++ Start Machine Learning... +++")

        # import testing data into a pandas data frame
        self.df = pd.read_csv('training_data.csv', header=0)

        self.X_data_complete = self.df.iloc[:,0:64].values         # iloc == "integer locations" of rows/cols
        self.y_data_complete = self.df['64'].values       # individually addressable columns (by name)
        
        self.X_known = self.X_data_complete[:,:]
        self.y_known = self.y_data_complete[:]
        
        self.KNOWN_SIZE = len(self.y_known)
        indices = np.random.permutation(self.KNOWN_SIZE)  # this scrambles the data each time
        self.X_known = self.X_known[indices]
        self.y_known = self.y_known[indices]

        TRAIN_FRACTION = 0.85
        TRAIN_SIZE = int(TRAIN_FRACTION*self.KNOWN_SIZE)
        self.TEST_SIZE = self.KNOWN_SIZE - TRAIN_SIZE   # not really needed, but...
        self.X_train = self.X_known[:TRAIN_SIZE]
        self.y_train = self.y_known[:TRAIN_SIZE]

        self.X_test = self.X_known[TRAIN_SIZE:]
        self.y_test = self.y_known[TRAIN_SIZE:]

        USE_SCALER = True
        if USE_SCALER == True:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(self.X_known)   # Fit only to the training dataframe
            # rescale inputs: both testing and training
            self.X_known = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            

        # write png to csv to create data set
        # train_csv = []

        # for i in range(200):
        #     train_csv += [png_to_csv(f'mask{i}.png')]
        #     train_csv += [png_to_csv2(f'face{i}.png')]
        

        # write_to_csv(train_csv, 'training_data.csv')

        print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")

        # train neural network
        self.mlp = MLPClassifier(hidden_layer_sizes=(128,100), max_iter=400, alpha=1e-6,
                            solver='adam', verbose=True, shuffle=True, early_stopping = False, tol=1e-4, 
                            random_state=None, # reproduceability
                            learning_rate_init=.003, learning_rate = 'adaptive')
        
        self.mlp.fit(self.X_train, self.y_train)


        print("\n\n++++++++++++  TESTING  +++++++++++++\n\n")
        print("Training set score: %f" % self.mlp.score(self.X_train, self.y_train))
        print("Test set score: %f" % self.mlp.score(self.X_test, self.y_test))


        # predictions:
        predictions = self.mlp.predict(self.X_test)
        print(predictions)

        # from sklearn.metrics import classification_report,confusion_matrix
        # print("\nConfusion matrix:")
        # print(confusion_matrix(y_test,predictions))

        # print("\nClassification report")
        # print(classification_report(y_test,predictions))

        # unknown data rows...
        #
        #unknown_predictions = mlp.predict(X_unknown)


        print("+++ End Machine Learning... +++")


    def detect_mask(self):
        """ member method to detect masks of user
        * input: none
        * displays image of user with result
        """

        print("+++ Start Mask Recognition... +++")

        # face_cascade = cv2.CascadeClassifier('/Users/wyattchang/Desktop/CS35_Final/haarcascade_frontalface_default.xml')

        # import face cascade classifier
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0) #Capture Live Video Feed

        c = 0
        
        # press s key to start
        if keyboard.is_pressed('s'):
            ret, img = cap.read()
            print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces =  face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces: 
                face = img[y:y+h,x:x+w]
                face = cv2.resize(face, (8,8), ) #resize to an 8x8 image
                #cv2.imwrite(f"mask{c}.png",face)  #cv2.savefile save to file
                cv2.imwrite("live.png",face)
                #cv2.imshow('img1',face)
                live_data = [png_to_csv("live.png")]
                
                write_to_csv(live_data,'live_data.csv')
                df1 = pd.read_csv('live_data.csv', header=None)

                X_data_live = df1.iloc[:,:64].values         # iloc == "integer locations" of rows/cols
                
                c += 1
                
                # # mlp = MLPClassifier(hidden_layer_sizes=(15,14,14,10), max_iter=400, alpha=1e-4,
                # #             solver='sgd', verbose=True, shuffle=True, early_stopping = False, tol=1e-4, 
                # #             random_state=None, # reproduceability
                # #             learning_rate_init=.03, learning_rate = 'adaptive')
                #mlp.fit(X_train, y_train)
                

                USE_SCALER = True
                if USE_SCALER == True:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    # scaler.fit(self.X_known)   # Fit only to the training dataframe
                    # # now, rescale inputs -- both testing and training
                    # self.X_known = scaler.transform(self.X_train)
                    # self.X_test = scaler.transform(self.X_test)
                    # #X_unknown = scaler.transform(X_unknown)
                    X_data_live = scaler.transform(X_data_live)
                prediction = self.mlp.predict(X_data_live)
                print(prediction)
                # print(type(prediction[0]))
                
                if prediction == [1]:
                    cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)
                if prediction == [0]:
                    cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),2)   
                cv2.imshow('img',img)
                
                
                # mask = mask_cascade.detectMutliScale(roi_gray)
                # for (mx,my,mw,mh) in mask:
                #     cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh),(0,255,0),2)

                

                cv2.imshow('img',img)
            if keyboard.is_pressed('s'):
                cap.release()
                cv2.destroyAllWindows()
        # break

net = NeuralNet()
net.detect_mask()
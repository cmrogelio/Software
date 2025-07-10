import os
#os.environ["KERAS_BACKEND"]="tensorflow"
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
from PyQt5 import QtCore, QtGui
from sklearn.metrics import confusion_matrix
import socket
# CVS libs
import csv

## Read data from scv files
def DataExtract(DataPath,type):
    #   Type    0 -> table or row
    #           1 -> column
    if(os.path.exists(DataPath)):   # Checks if the file exists
        Table=[]                    # Table to store the data
        ## Initialization of the document reading
        with open(DataPath,'r', encoding='utf-8-sig') as csvFile:   # Open the file
            Lines = csvFile.readlines()                             # Read all the lines on the file
            for line in Lines:                                      # Iterate over each of the lines 
                splitLine=line.split(',')                           # Split the line by the comma (,) in to an array
                Row=[]                                              # Array to store the elements of the current line  
                for item in splitLine:                              # Iterate over each of the elements in the line
                    if(item!='/n'):                                 # Checks if the element is not empty
                        item = item.split("\n")[0]                  # Remove the \n character
                        if(item != ''):                             # Checks if the element is not empty
                            if(item[0:3] == 'ï»¿'):                 # Remove the special character added by the excel
                                item = item[3:]         
                            if(type):
                                Row.append(item)                    # If the type is 1, the data is stored in a column format
                            else:
                                Row = item                          # If the type is 0, the data is stored in a row format
                Table.append(Row)                                   # Adds the rows in a single array
            Data = np.array(Table)                                  # Converts the array to a numpy array
        return Data                                                 # Returns the data
    else:
        return 0                                                    # If the file doesn't exist, return 0

##  Basic Sequential model
##   This model is used for classification problems, the data must be normalized and the output must be a one-hot vector
def buildSequentialModel(iSize=17,hSize=[64,64],drop=True,dRate=0.25,classes=2,lr=0.001):
    modelSequential = keras.models.Sequential()                                      # Model initialization
    modelSequential.add(keras.layers.Input(shape=(iSize,)))                          # Adds the Input layer, the shape defines the number of features for each sample
    for layer in hSize:                                                              # Adds the hidden layers, the number of layers is defined by the hSize array
        modelSequential.add(keras.layers.Dense(units=layer, activation='relu'))      # Adds the hidden layer, the number of units is defined by each element of the hSize array and the activation function ReLU is used
    if(drop):                                                                        # Checks if the Dropout layer is needed
        modelSequential.add(keras.layers.Dropout(rate=dRate))                        # The Dropout layer allow to turnoff neurons randomly to avoid overfitting, the rate is defined by the dRate variable
    modelSequential.add(keras.layers.Dense(units=classes, activation='softmax'))     # Adds the output layer, the number of units is defined by the classes variable and softmax is usued to classify the data
    modelSequential.compile(                                                         # Defines the model compilation parameters
    loss=keras.losses.SparseCategoricalCrossentropy(),                               # Loss function, the SparseCategoricalCrossentropy is used for multiclass classification problems
    optimizer=keras.optimizers.Adam(learning_rate=lr),                               # Optimizer, Adam is used with a learning rate defined by the lr variable
    metrics=["accuracy"],                                                            # Metrics, the accuracy is used to measure the performance of the model
    )

    modelSequential.summary()                                                           # Generates a table with the basic structure of the model
    keras.utils.plot_model(modelSequential, "Sequiential_model.png", show_shapes=True)  # Generates a more detail figure, with the sizes of the data in each layer
    return modelSequential                                                              # Returns the model

##  LSTM for prediction
##   This model is used for prediction problems, the data must be normalized and the output must be a single value
##   The LSTM model is used for time series prediction, the data must be reshaped to be used in the LSTM model using the formatLSTM function
def buildLSTMModel(wSize=5,iSize=17,hSize=[64,64],lr=0.001):
    inputLayer = keras.Input(shape=(wSize,iSize))                                                   # Input layer, the shape is defined by the wSize (window size) and iSize (no. of features) and can take any number of samples
    LSTMlayer = keras.layers.LSTM(units=hSize.pop(0))(inputLayer)                                   # Defines a LSTM layer with the inputLayer as its input, the number of units is defined by the first element of the hSize array
    denselayer = keras.layers.Dense(units=hSize.pop(0),activation='relu')(LSTMlayer)                # Defines a Dense layer with the LSTM layer as its input, the number of units is defined by the second element of the hSize array and ReLU is used as activation function
    outputLayer = keras.layers.Dense(units=iSize, activation='relu')(denselayer)                    # Defines the output layer, the number of units is equal to the number of features since the output id the next value of the time series
    modelFunctional = keras.Model(inputs=inputLayer, outputs=outputLayer, name="Test_LSTM_model")   # Defines the model, the input and output layers are defined by the inputLayer and outputLayer variables respectively

    modelFunctional.compile(                                                                        # Defines the model compilation parameters
    loss=keras.losses.MeanSquaredError(),                                                           # Loss function, the MeanSquaredError is used for regression problems
    optimizer=keras.optimizers.Adam(learning_rate=lr),                                              # Optimizer, Adam is used with a learning rate defined by the lr variable
    metrics=[keras.metrics.RootMeanSquaredError()]                                                  # Metrics, the RootMeanSquaredError is used to measure the performance of the model
    )

    modelFunctional.summary()                                                                       # Generates a table with the basic structure of the model
    keras.utils.plot_model(modelFunctional, "fuctional_model_shape_info.png", show_shapes=True)     # Generates a more detail figure, with the sizes of the data in each layer

    return modelFunctional                                                                          # Returns the model

##  LSTM for classification
##   This model is used for classification problems, the data must be normalized and the output must be a one-hot vector
##   The LSTM model is used for time series classification, the data must be reshaped to be used in the LSTM model using the formatLSTM function
def buildLSTMClassificationModel(wSize=5,iSize=17,hSize=[64,64],classes=2,lr=0.001,imagePath="M"):
    inputLayer = keras.Input(shape=(wSize,iSize))                                                           # Input layer, the shape is defined by the wSize (window size) and iSize (no. of features) and can take any number of samples
    LSTMlayer = keras.layers.LSTM(units=hSize.pop(0))(inputLayer)                                           # Defines a LSTM layer with the inputLayer as its input, the number of units is defined by the first element of the hSize array
    hiddenLayer = keras.layers.Dense(units=hSize.pop(0),activation='relu')(LSTMlayer)                       # Defines a Dense layer with the LSTM layer as its input, the number of units is defined by the second element of the hSize array
    for value in hSize:                                                                                     # Iterate over the hSize array to add the hidden layers
        hiddenLayer = keras.layers.Dense(units=value,activation='relu')(hiddenLayer)                        # Adds the hidden layer, the number of units is defined by each element of the hSize array
    outputLayer = keras.layers.Dense(units=classes, activation='softmax')(hiddenLayer)                      # Defines the output layer, softmax is usued to classify the data

    modelFunctional = keras.Model(inputs=inputLayer, outputs=outputLayer, name="Classification_LSTM_model") # Defines the model, the input and output layers are defined by the inputLayer and outputLayer variables respectively

    modelFunctional.summary()                                                                               # Generates a table with the basic structure of the model
    imagePath += '-Shape.png'
    keras.utils.plot_model(modelFunctional, imagePath, show_shapes=True)                                    # Generates a more detail figure, with the sizes of the data in each layer this is save in the model path

    modelFunctional.compile(                                                                                # Defines the model compilation parameters
    loss=keras.losses.SparseCategoricalCrossentropy(),                                                      # Loss function, the SparseCategoricalCrossentropy is used for multiclass classification problems
    optimizer=keras.optimizers.RMSprop(),                                                                   # Optimizer, RMSprop is used with a learning rate defined by the lr variable
    metrics=["accuracy"],                                                                                   # Metrics, the accuracy is used to measure the performance of the model
    )

    return modelFunctional                                                                                  # Returns the model

##  Reshapes the data to be used in the LSTM model
##  The data must be reshaped to be used in the LSTM model since the LSTM model requires a window of data to make a prediction or classification
def formatLSTM(wSize=5,data=[]):            # wSize is the window size, data is the data to be reshaped
    inputData = []
    for i in range(wSize, len(data)+1):     # Iterate over the data to create the windows
        inputSample = data[i - wSize:i]     # Create the window of data
        inputData.append(inputSample)       # Append the window to the inputData array
    inputData = np.array(inputData)         # Convert the inputData array to a numpy array
    return inputData                        # Return the reshaped data, this is a trople array with the shape (samples, window size, features)

##  Reshapes the data to be used in the LSTM model
##  The data must be reshaped to be used in the LSTM model, the difference with the formatLSTM function is that this function adds padding to the data
##  this padding means that the window of time is not generated with consecutive data of the time series, instead it takes samples separated by a padding value of time
##  Example: if array = [1,2,3,4,5,6] and wSize = 3 and padding = 2, the output will be [[1,3,5],[2,4,6]]
def formatLSTM2(wSize=5,data=[],padding=10):
    inputData = []
    for i in range((wSize-1)*padding, len(data)):           # Iterate over the data to create the windows, starting from the (wSize-1)*padding value
        inputSample = []
        for j in range((wSize-1),-1,-1):                    # Iterate over the time series to create the windows, by checking the previous values of the time series
            index = i-(padding*j)
            inputSample.append(data[index])                 # Append the value of the time series to the inputSample array
        inputData.append(inputSample)                       # Append the window to the inputData array
    inputData = np.array(inputData)                         # Convert the inputData array to a numpy array
    return inputData                            # Return the reshaped data, this is a trople array with the shape (samples, window size, features)

##  Basic Functional model
##   This model is used for classification problems, the data must be normalized and the output must be a one-hot vector
def buildFunctionalModel(iSize=17,hSize=[64,64],drop=True,dRate=0.25,classes=2,lr=0.001):
    inputLayer = keras.Input(shape=(iSize,))                                                    # Input layer, the shape is defined by the iSize (no. of features) and can take any number of samples 
    hiddenLayer = keras.layers.Dense(units=hSize.pop(0),activation='relu')(inputLayer)          # First layer, this layer has a fixed input that is the inputLayer
    for value in hSize:                                                                         # Iterate over the hSize array to add the hidden layers
        hiddenLayer = keras.layers.Dense(units=value,activation='relu')(hiddenLayer)            # Adds the hidden layer, the number of units is defined by each element of the hSize array
    outputLayer = keras.layers.Dense(units=classes, activation='softmax')(hiddenLayer)          # Defines the output layer, softmax is usued to classify the data
                                                                                                # The classifier and loss function must be selected carefully because the output, the desire output and the loss function must have the same format
    modelFunctional = keras.Model(inputs=inputLayer, outputs=outputLayer, name="Test_model")    # Define the model input and output layers are defined by the inputLayer and outputLayer variables respectively

    modelFunctional.summary()                                                                   # Generates a table with the basic structure of the model
    keras.utils.plot_model(modelFunctional, "fuctional_model_shape_info.png", show_shapes=True) # Generates a more detail image of the model conections

    #   Every model need to be compile after it's fully structured, this is done to define the loss function and the optimizer
    modelFunctional.compile(                                                                    # Defines the model compilation parameters
    loss=keras.losses.SparseCategoricalCrossentropy(),                                          # Loss function, the SparseCategoricalCrossentropy is used for multiclass classification problems
    optimizer=keras.optimizers.RMSprop(),                                                       # Optimizer, RMSprop is used with a learning rate defined by the lr variable
    metrics=["accuracy"],                                                                       # Metrics, the accuracy is used to measure the performance of the model
    )

    return modelFunctional                                                                      # The compile model is returned

##  Build the Confusion matrix plot
##  This function is used to plot the confusion matrix, this is only the visual part of the confusion matrix, the data must be generated using the confusion_matrix function from sklearn.metrics
def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues, save = True, path = './'):

    #   cm is the confusion matrix, it can be generated using:
    #       cm = confusion_matrix(y_true=Ground_truth, y_pred=Predicted_values)
    #   This is a function that must be imported from:
    #       from sklearn.metrics import confusion_matrix

    plt.imshow(cm, interpolation='nearest',cmap=cmap)                           # Generates the plot of the confusion matrix
    plt.title(title)                                                            # Assigns the title to the plot
    plt.colorbar()                                                              # Adds a color bar to the plot
    tick_marks = np.arange(len(classes))                                        # Defines the number of division for the class labels
    plt.xticks(tick_marks,classes, rotation=45)                                 # Adds the labels in the X axis with a slight rotation
    plt.yticks(tick_marks, classes)                                             # Adds the labels in the Y axis

    if normalize:                                                               # Checks if the confusion matrix wants to be normalized   
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]                  # Normalizes the confusion matrix by dividing each element by the sum of the elements in the row
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normailization')

    thresh = cm.max()/2.                                                        # Defines a threshold for the color of the font (white if it's greater than the threshold and black otherwise)
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):       # Iterate over the confusion matrix to add the values in each cell
        plt.text(j, i, cm[i, j],horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")      # Adds the value in each cell of the confusion matrix
    
    plt.tight_layout()                                                          # Allow the adjusment of the plot to the form's shape
    plt.ylabel('True label')                                                    # Y axis label
    plt.xlabel('Predicted label')                                               # X axis label

    if(save):
        name = path + '-ConfusionMatrix.png'                                    # Checks if the plot wants to be save 
        plt.savefig(name, dpi=300)                                              # Saves the plot as a figure
    plt.show()                                                                  # Show the plot

##  Draw keypoints
##  This function is used to draw the keypoints in the image, this is used to visualize the keypoints detected by the model
def drawKeyPoints(frame,keypoints):
    #   frame: must be an array of images, at leat one
    #   keypoints: must be an array of keypoints one for each frame
    blank = np.zeros((300,300,3))                           # Creates a blank image to draw the keypoints
    frameKey = [blank,blank]                                # Creates an array of blank images to draw the keypoints
    for f in range(len(keypoints)):                         # Iterate over the frames to draw the keypoints
        for point in keypoints[f]:
            if(np.max(point) > 0):                          # Checks if the keypoint is valid, when the keypoint is not valid the value in all axis is 0
                cv2.ellipse(frame[f], (int(point[0]),int(point[1])), (4,4),0,0,360,(255,0,0),4)             # Draws the keypoint in the image
        frameKey[f] = np.copy(frame[f])                     # Copies the image with the keypoints drawn to the frameKey array

##   Change the format of the image to be display in the UI
def imageFormat(originalImg):
    frame = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)                                                        # Converts the image from BGR to RGB format     
    image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)   # Converts the image to a QImage format 
    newImg = QtGui.QPixmap.fromImage(image)                                                                     # Converts the QImage to a QPixmap format to be used in the UI
    return newImg                                                                                               # Returns the QPixmap image

##  This function is used to detect the skeleton in the image using the YOLO model, this is used to detect the keypoints in the image
def skeletonDetection(frame,threshold,model):
    blank = (np.zeros((300,300,3))).astype('uint8')                         # Creates a blank image to draw the keypoints
    frameYOLO = np.copy([blank,blank])                                      # Creates an array of blank images to draw the keypoints    
    keypoints = [[],[]]                                                     # Creates an array to store the keypoints detected by the model
    frame_height = [0,0]                                                    # Creates an array to store the height of the bounding box detected by the model
    Cam_origin = [[],[]]                                                    # Creates an array to store the origin that is going to be used to transform the coordinates of the keypoints
    for j in range(len(frame)):
        results = model(frame[j], conf=threshold, verbose=False)[0]         # Detects the keypoints in the image using the YOLO model
        frameYOLO[j] = np.copy(results.plot())                              # Draws the bounding box and the keypoints in the image
        keypoints[j] = results.keypoints.numpy().xy[0]                      # Gets the keypoints coordinates detected by the model
        if(len(keypoints[j])==0):                                           # Checks if the keypoints are valid, when the keypoints are not valid the value in all axis is 0
            keypoints[j] = np.copy((np.zeros((17,2))).astype('float32'))    # If the keypoints are not valid, the keypoints are set to 0
        if(len(results.boxes.data.tolist()) != 0):                          # Checks if the bounding box is valid, when the bounding box is not valid the value in all axis is 0
            box = results.boxes.data.tolist()[0]                            # Gets the bounding box coordinates detected by the model
            x1, y1, x2, y2, score, class_id = box                           # Gets the coordinates of the bounding box
            frame_height[j] = y2-y1                                         # Gets the height of the bounding box
            Cam_origin[j] = [x1,y2]                                         # Gets the origin of the bounding box
    
    return [frameYOLO,keypoints,frame_height,Cam_origin]                    # Returns the image with the bounding box and the keypoints drawn, the keypoints detected by the model, the height of the bounding box and the origin of the bounding box

## This function is used to stablish a serial comunication with a C++ program, this is used to send and receive data
def comunication():
    HOST = '127.0.0.1'                                                  # The server's hostname or IP address
    PORT = 65432                                                        # The port used by the server

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)               # Create a socket object
    s.bind((HOST, PORT))                                                # Bind the socket to the host and port
    s.listen()                                                          # Listen for incoming connections

    conn, addr = s.accept()                                             # Accept the connection from the client
    # print(f"Connect by {addr}")                                       # Print the address of the client

    return conn                                                         # Return the connection object

## This function is used to filter the data using an Exponential Moving Average (EMA) filter
## The EMA filter is used to smooth the data and remove noise from the data
def filterEMA(data=[],x=0.5):
    #   x is the smoothing factor, the value must be between 0 and 1, the closer to 1 the more smooth the data is
    newData = []
    for i in range(len(data[0])):                                                   # Checks the data column by column
        movingAveranges = []                                                        # Creates an array to store the moving averages
        movingAveranges.append(data[0][i])                                          # The first value is the same as the original data, but this will be removed later
        for j in range(len(data)):                                                  # Checks each row of the column
            windowAverage = round((x*data[j][i])+(1-x)*movingAveranges[-1], 2)      # Calculates the moving average using the EMA formula
            movingAveranges.append(windowAverage)                                   # Appends the moving average to the movingAveranges array
        newData.append(movingAveranges[1:])                                         # Appends the moving averages to the newData array as a row, removing the first value since it is the same as the original data
    
    newData = np.array(newData)                                                     # Converts the newData array to a numpy array
    newData = np.transpose(newData)                                                 # Transposes the newData changing the shape from (rows,columns) to (columns,rows)

    return newData                                                                  # Returns the newData array with the moving averages

## This function is used to interpolate the data using the numpy interp function
## The interpolation is used to fill the missing values in the data
def intepolate(data=[], time=[]):
    for i in range(len(data[0])):                                               # Checks the data column by column
        search = []                                                             # Creates an array to store the time values of the missing data
        index = []                                                              # Creates an array to store the indexes of the missing values
        xp = []                                                                 # Creates an array to store the time values of the valid data
        fp = []                                                                 # Creates an array to store the valid data
        for j in range(len(data)):                                              # Checks each row of the column
            if(data[j][i]==0):                                                  # Checks if the value is missing, meaning that the value is 0
                search.append(time[j])                                          # Appends the time value to the search array
                index.append(j)                                                 # Appends the index of the missing value to the index array
            else:
                fp.append(data[j][i])                                           # Appends the valid data to the fp array
                xp.append(time[j])                                              # Appends the time value to the xp array

        if(len(xp) >0 and len(fp) > 0 and len(search) > 0):                     # Checks if there are valid data to interpolate
            res = np.interp(search, xp, fp)                                     # Interpolates the missing values using the numpy interp function
            for k in range(len(index)):                                         # Checks the indexes of the missing values
                data[index[k]][i] = res[k]                                      # Assigns the interpolated value to the missing value in the data array

    return data                                                                 # Returns the data array with the interpolated values

## This function is used to calculate the angle between two points in the keypoints array
def angle(keypoints, points = [6,12], acute=True):
    arg1 = keypoints[points[0]][2] - keypoints[points[1]][2]            # z0-z1
    arg2 = keypoints[points[0]][0] - keypoints[points[1]][0]            # x0-x1
    theta = np.arctan2(arg1,arg2)*(180/np.pi)                           # Calculates the angle between the two points in degrees
                                                                        # theta = atan((z0-z1)/(x0-x1))
    return theta                                                        # Returns the angle in degrees

## This function takes the array of 3d keypoints and converts it to a row format, this is used to store the data in a single row
def poseRow(keypoints3d):
    keypointsRow = np.zeros(len(keypoints3d)*3)                 # Creates an array to store the keypoints in a row format
    for i in range(len(keypoints3d)):                           # Checks each of the keypoints
        keypointsRow[(i*3)] = keypoints3d[i][0]                 # Assigns the x coordinate of the keypoint to the row array
        keypointsRow[(i*3)+1] = keypoints3d[i][1]               # Assigns the y coordinate of the keypoint to the row array
        keypointsRow[(i*3)+2] = keypoints3d[i][2]               # Assigns the z coordinate of the keypoint to the row array
    keypointsRow = np.array(keypointsRow)                       # Converts the keypointsRow array to a numpy array

    return keypointsRow                                         # Returns the keypoints in a row format

## This function takes the keypoints of both cameras and converts them to a 3D format
def Pose3D(keypoints,frame_height,Cam_origin):
    for i in range(len(keypoints[0])):                          # Checks each of the keypoints
        if(min(keypoints[0][i])!= 0):                           # Checks if the keypoint is valid, when the keypoint is not valid the value in all axis is 0
            keypoints[0][i] = keypoints[0][i]-Cam_origin[0]     # To the keypoints coordinates of the camera A frame is substracted the new origin obtained from the bounding box

        if(min(keypoints[1][i])!= 0):                           # Checks if the keypoint is valid, when the keypoint is not valid the value in all axis is 0
            keypoints[1][i] = keypoints[1][i]-Cam_origin[1]     # To the keypoints coordinates of the camera B frame is substracted the new origin obtained from the bounding box              

    if(min(frame_height)>0):                                    # Checks if the height of the bounding box is valid, when the height is not valid the value in all axis is 0
        if(frame_height[0] < frame_height[1]):                  # Checks if the height of the bounding box from camera A is less than the height of the bounding box from camera B
            scale = frame_height[1]/frame_height[0]             # Calculates the scale to be used to transform the coordinates of the keypoints
            keypoints[0] = keypoints[0]*scale                   # Transforms the coordinates of the keypoints from camera A to be adjusted to the height of the bounding box from camera B
        if(frame_height[1] < frame_height[0]):                  # Checks if the height of the bounding box from camera B is less than the height of the bounding box from camera A
            scale = frame_height[0]/frame_height[1]             # Calculates the scale to be used to transform the coordinates of the keypoints
            keypoints[1] = keypoints[1]*scale                   # Transforms the coordinates of the keypoints from camera B to be adjusted to the height of the bounding box from camera A

    keypoints3D = []                                            # Creates an array to store the keypoints in a 3D format
    hs = 1/188  #(2.412/300)                                            # Height scale to be used to transform the coordinates of the keypoints from pixels to meters
    vs = -1/188  #(-1.513/300)                                           # Width scale to be used to transform the coordinates of the keypoints from pixels to meters
    for i in range(len(keypoints[0])):                          # Checks each of the keypoints
        # Points format (x,y,z,z2)
        # x = x coordinate of the keypoint from camera A
        # y = x coordinate of the keypoint from camera B
        # z = y coordinate of the keypoint from camera A
        # z2 = y coordinate of the keypoint from camera B, but this is not used in the 3D format
        Point = [keypoints[0][i][0]*hs,keypoints[1][i][0]*hs,keypoints[0][i][1]*vs, keypoints[1][i][1]*vs]
        keypoints3D.append(Point)                               # Appends the keypoint to the keypoints3D array

    return keypoints3D                                         # Returns the keypoints in a 3D format

def dataFileInitializer(mainPath,name,termination):
    dataPath = mainPath + '/' + name + '-' + termination                           # Creates the path for the data file
    with open(dataPath,'w',newline='',encoding='utf-8-sig') as csv_file:
        Tags = ["Time","Hour"]
        info = {}
        info["Time"]="Time"
        info["Hour"]="Hour"
        for i in range(17):
            Tag = [f"X{i}",f"Y{i}",f"Z{i}"]
            Tags += Tag
            info[Tag[0]]=Tag[0]
            info[Tag[1]]=Tag[1]
            info[Tag[2]]=Tag[2]
        csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
        csv_writer.writeheader
        csv_writer.writerow(info)
    return dataPath
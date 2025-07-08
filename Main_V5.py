# UI design lib
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
# Oak-D libs
import depthai as dai
import blobconverter
# Theading lib
import threading
import contextlib
# Artificial vision lib
import cv2
import numpy as np
# YOLO libs
from ultralytics import YOLO
# CVS libs
import csv
# Time lib
import datetime
# Operative system
import os
# Local lib
import ML_V5 as ML
# Normalice lib
from sklearn.preprocessing import MinMaxScaler
# Machine learning lib
import keras


class Ui_MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Ui_MainWindow, self).__init__(*args, **kwargs)    # Defines the Form as the main window
        self.initialVariables()                                 # Variables initialization  
        self.setupUi()                                          # UI setup 
    
    # Function to initialize the project variables
    def initialVariables(self):
        # File names
        self.mainPath = "./RecordedData/"                       # Main path for the video recording
        self.fileData = 'Movement.csv'                          # File name for the live movement data
        self.fileScaler = 'Scaler.csv'                          # File name for the scaler data
        self.fileTime = 'Time.csv'                              # File name for the time data
        self.fileArray = 'LastPose.csv'                         # File name for the movement data of the last aquired frame
        # Video camera parameters
        self.fps = 30                                           # Frames per second, this is define by the camera specification
        self.widthVideo = 300                                   # Width of the video frame
        self.heightVideo = 300                                  # Height of the video frame
        blank = np.zeros((self.widthVideo,self.heightVideo,3))  # Blank image
        self.blank = blank.astype('uint8')                      # Change the blank image in uint8 format
        self.frame = [self.blank,self.blank]                    # Array to store the two cameras frames, initialized with the blank image
        self.frameKey = np.copy(self.frame)                     # Array to store the key points frames with the keypoints drawned
        self.frameYOLO = np.copy(self.frame)                    # Array to store the YOLO frames with the skeleton drawned
        self.frameCount = 0                                     # Frame counter
        # Flags
        self.videoFlag = 0                                      # Video flag, 0 -> Raw video, 1 -> Skeleton detection, 2 -> Key points detection
        self.snapFlag = 0                                       # Snap flag, 0 -> No live image processing, 1 -> Live image processing
        self.timerFlag = 0                                      # Timer flag, 0 -> No timer, 1 -> Starts recording and start timer
        self.closeFlag = 0                                      # Close flag, 0 -> No close, 1 -> The program is closing
        self.STSFlag = 0                                        # Sit-to-stand flag, 0 -> No STS test, 1 -> STS test
        self.modelSTSFlag = 0                                   # Model flag, 0 -> No model applied, 1 -> Model application during live video
        self.commFlag = 0                                       # Communication flag, 0 -> No communication, 1 -> Search to stablish a serial communication
        # Image processing variables
        blankPoints = np.zeros((17,2))                          # Blank array with the same shape of the keypoints
        self.blankPoints = blankPoints.astype('float32')        # Change the blank image to float32 format
        self.keypoints = [self.blankPoints,self.blankPoints]    # Array to store the keypoints of the two cameras for the latest frame
        self.frame_height = [[],[]]                             # Array to store the height of the bounding box for each of the cameras frames
        self.Cam_origin = [[],[]]                               # Array to store the origin of the bounding box for each of the cameras frames
        # YOLO parameters
        modelSize = 1                                           # (1-5) While using live video the value must always be 1
        YOLO_models = {1:'yolov8n-pose.pt',2:'yolov8s-pose.pt',3:'yolov8m-pose.pt',4:'yolov8l-pose.pt',5:'yolov8x-pose.pt'} # YOLO models
        self.model = YOLO(YOLO_models[modelSize])               # Load the YOLO model
        self.timeFeatures = []                                  # Array to store the time features for the sit to stand (STS) model application
        self.timeFeaturesFall = []                              # Array to store the time features for the fall risk model application

    # Function to initialize the UI elements
    # Most of the UI elements are created using the Qt Designer, so the code is generated automatically
    # The only changes in the code are the connections to the functions, in the case of the buttons and radio buttons
    def setupUi(self):
        ## Main window parameters
        self.setObjectName("MainWindow")                                # Defines the name of the main window
        self.resize(1500, 650)                                          # Size of the main window
        self.centralwidget = QWidget(self)                              # Defines that the central widget will be the main window
        self.centralwidget.setObjectName("centralwidget")               # Defines the name of the central widget
        
        ## Display the image of the camera A
        ## This is made by creating a label and setting the image as a pixmap, since the PyQt5 library does not have an object to display images directly
        self.PictureBoxA = QLabel(self.centralwidget)                       # Defines the name of the label that will display the image of the camera A
        self.PictureBoxA.setGeometry(QtCore.QRect(30, 30, 600, 600))        # Sets the position and size of the PictureBoxA
        self.PictureBoxA.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))    # Change the cursor to a cross cursor when hovering over the PictureBoxA
        self.PictureBoxA.setMouseTracking(True)                             # Enables mouse tracking for the PictureBoxA
        self.PictureBoxA.setFrameShape(QFrame.Panel)                        # Sets the frame shape of the PictureBoxA to a panel
        self.PictureBoxA.setLineWidth(2)                                    # Sets the line width of the PictureBoxA to 2 pixels
        self.PictureBoxA.setAutoFillBackground(True)                        # Enables auto fill background for the PictureBoxA
        self.PictureBoxA.setText("")                                        # Sets the text of the PictureBoxA to an empty string, since is used to display an image not a text
        self.PictureBoxA.setPixmap(QtGui.QPixmap("Holder.png"))             # Sets the pixmap of the PictureBoxA to a placeholder image
        self.PictureBoxA.setScaledContents(True)                            # Enables scaling of the image to fit the PictureBoxA size
        self.PictureBoxA.setWordWrap(False)                                 # Disables word wrap for the PictureBoxA, to avoid text wrapping
        self.PictureBoxA.setObjectName("PictureBoxA")                       # Sets the name of the PictureBoxA

        self.PictureBoxB = QLabel(self.centralwidget)
        self.PictureBoxB.setGeometry(QtCore.QRect(670, 30, 600, 600))
        self.PictureBoxB.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.PictureBoxB.setMouseTracking(True)
        self.PictureBoxB.setFrameShape(QFrame.Panel)
        self.PictureBoxB.setLineWidth(2)
        self.PictureBoxB.setAutoFillBackground(True)
        self.PictureBoxB.setText("")
        self.PictureBoxB.setPixmap(QtGui.QPixmap("Holder.png"))
        self.PictureBoxB.setScaledContents(True)
        self.PictureBoxB.setWordWrap(False)
        self.PictureBoxB.setObjectName("PictureBoxB")

        ## Picture box for STS Test
        self.PictureBoxC = QLabel(self.centralwidget)
        self.PictureBoxC.setGeometry(QtCore.QRect(550, 230, 200, 200))
        self.PictureBoxC.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.PictureBoxC.setMouseTracking(True)
        self.PictureBoxC.setFrameShape(QFrame.Panel)
        self.PictureBoxC.setLineWidth(2)
        self.PictureBoxC.setAutoFillBackground(True)
        self.PictureBoxC.setText("None")
        self.PictureBoxC.setObjectName("PictureBoxC")
        self.PictureBoxC.setStyleSheet("background-color : white")
        self.PictureBoxC.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(30)
        self.PictureBoxC.setFont(font)
        self.PictureBoxC.setVisible(False)

        self.PictureBoxD = QLabel(self.centralwidget)
        self.PictureBoxD.setGeometry(QtCore.QRect(550, 400, 200, 200))
        self.PictureBoxD.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.PictureBoxD.setMouseTracking(True)
        self.PictureBoxD.setFrameShape(QFrame.Panel)
        self.PictureBoxD.setLineWidth(2)
        self.PictureBoxD.setAutoFillBackground(True)
        self.PictureBoxD.setText("None")
        self.PictureBoxD.setObjectName("PictureBoxD")
        self.PictureBoxD.setStyleSheet("background-color : white")
        self.PictureBoxD.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(30)
        self.PictureBoxD.setFont(font)
        self.PictureBoxD.setVisible(False)

        ## Start button
        self.BtnStart = QPushButton(self.centralwidget)
        self.BtnStart.setGeometry(QtCore.QRect(1310, 450, 181, 41))
        self.BtnStart.setObjectName("BtnStart")
        self.BtnStart.clicked.connect(self.StartRecord)

        ## Sit-to-stand button
        self.BtnSTS = QPushButton(self.centralwidget)
        self.BtnSTS.setGeometry(QtCore.QRect(1310, 510, 181, 41))
        self.BtnSTS.setObjectName("BtnSTS")
        self.BtnSTS.clicked.connect(self.STSTest)

        ## Model button
        self.BtnModel = QPushButton(self.centralwidget)
        self.BtnModel.setGeometry(QtCore.QRect(1310, 570, 181, 41))
        self.BtnModel.setObjectName("BtnModel")
        self.BtnModel.clicked.connect(self.monitorVideo)

        ## Read button
        self.BtnRead = QPushButton(self.centralwidget)
        self.BtnRead.setGeometry(QtCore.QRect(1310, 570, 181, 41))
        self.BtnRead.setObjectName("BtnRead")
        self.BtnRead.setVisible(False)

        ## Group box
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1310, 30, 181, 151))
        self.groupBox.setObjectName("groupBox")
        font = QtGui.QFont()
        font.setBold(True)
        self.groupBox.setFont(font)

        ## Radio button Raw
        self.RbRaw = QRadioButton(self.groupBox)
        self.RbRaw.setGeometry(QtCore.QRect(20, 30, 150, 25))
        self.RbRaw.setObjectName("radioButton")
        self.RbRaw.setChecked(True)
        font = QtGui.QFont()
        font.setBold(False)
        self.RbRaw.setFont(font)

        ## Radio buton skeleton
        self.RbKey = QRadioButton(self.groupBox)
        self.RbKey.setGeometry(QtCore.QRect(20, 110, 150, 25))
        self.RbKey.setObjectName("radioButton_2")
        font = QtGui.QFont()
        font.setBold(False)
        self.RbKey.setFont(font)

        ## Radio buton key points
        self.RbTracking = QRadioButton(self.groupBox)
        self.RbTracking.setGeometry(QtCore.QRect(20, 70, 150, 25))
        self.RbTracking.setObjectName("radioButton_3")
        font = QtGui.QFont()
        font.setBold(False)
        self.RbTracking.setFont(font)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 461, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        ## Time label
        self.LbTime = QLabel(self.centralwidget)
        self.LbTime.setGeometry(QtCore.QRect(1310, 210, 111, 21))
        self.LbTime.setObjectName("LbTime")

        ## Time display label
        self.LbTimeDisplay = QLabel(self.centralwidget)
        self.LbTimeDisplay.setGeometry(QtCore.QRect(1310, 210, 181, 41))
        self.LbTimeDisplay.setObjectName("LbTime")
        self.LbTimeDisplay.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(15)
        self.LbTimeDisplay.setFont(font)
        self.LbTimeDisplay.setVisible(False)

        ## Time text box
        self.TbTime = QLineEdit(self.centralwidget)
        self.TbTime.setGeometry(QtCore.QRect(1310, 240, 181, 31))
        self.TbTime.setObjectName("TbTime")

        ## Name label
        self.LbName = QLabel(self.centralwidget)
        self.LbName.setGeometry(QtCore.QRect(1310, 280, 111, 21))
        self.LbName.setObjectName("LbName")

        ## Name text box
        self.TbName = QLineEdit(self.centralwidget)
        self.TbName.setGeometry(QtCore.QRect(1310, 310, 181, 31))
        self.TbName.setObjectName("TbName")

        # Check a radio button status
        #print(self.radioButton.isChecked())
        self.RbRaw.clicked.connect(self.visualizationChange)
        self.RbKey.clicked.connect(self.visualizationChange)
        self.RbTracking.clicked.connect(self.visualizationChange)

        # Timer
        self.timer = QTimer()
		# adding action to timer
        self.timer.timeout.connect(self.snap)
		# update the timer every tenth second
        self.timePerod = 10
        self.initialtime = 0
        self.goaltime = 0
        self.timer.start(10)

        # Process enable
        # Group box
        self.GbProcess = QGroupBox(self.centralwidget)
        self.GbProcess.setGeometry(QtCore.QRect(1310, 350, 181, 80))
        self.GbProcess.setObjectName("GbProcess")
        
        # Check box
        self.CbProcess = QCheckBox(self.GbProcess)
        self.CbProcess.setGeometry(QtCore.QRect(20, 30, 90, 25))
        self.CbProcess.setObjectName("CbProcess")
        self.CbProcess.setChecked(True)

        # Timer process
        self.timerProcess = QTimer()
        self.timerProcess.timeout.connect(self.timerAnalize)
        self.timerProcess.start(300)

        ## Hip Angle label
        self.LbHip = QLabel(self.centralwidget)
        self.LbHip.setGeometry(QtCore.QRect(1310, 280, 111, 21))
        self.LbHip.setObjectName("LbHip")

        ## Hip text box
        self.TbHip = QLineEdit(self.centralwidget)
        self.TbHip.setGeometry(QtCore.QRect(1310, 310, 181, 31))
        self.TbHip.setObjectName("TbName")
        self.TbHip.setVisible(False)

        ## Knee Angle label
        self.LbKnee = QLabel(self.centralwidget)
        self.LbKnee.setGeometry(QtCore.QRect(1310, 370, 111, 21))
        self.LbKnee.setObjectName("LbKnee")
        self.LbKnee.setVisible(False)

        ## Knee text box
        self.TbKnee = QLineEdit(self.centralwidget)
        self.TbKnee.setGeometry(QtCore.QRect(1310, 400, 181, 31))
        self.TbKnee.setObjectName("TbName")
        self.TbKnee.setVisible(False)

        ## Ankle Angle label
        self.LbAnkle = QLabel(self.centralwidget)
        self.LbAnkle.setGeometry(QtCore.QRect(1310, 460, 111, 21))
        self.LbAnkle.setObjectName("LbAnkle")
        self.LbAnkle.setVisible(False)

        ## Ankle text box
        self.TbAnkle = QLineEdit(self.centralwidget)
        self.TbAnkle.setGeometry(QtCore.QRect(1310, 490, 181, 31))
        self.TbAnkle.setObjectName("TbName")
        self.TbAnkle.setVisible(False)

        ## Process textbox
        self.TbProcess = QLabel(self)
        self.TbProcess.setGeometry(QtCore.QRect(1310, 30, 181, 150))
        self.TbProcess.setFrameShape(QFrame.Panel)
        self.TbProcess.setLineWidth(2)
        self.TbProcess.setAutoFillBackground(True)
        self.TbProcess.setText("Labeling")
        self.TbProcess.setObjectName("TbProcess")
        self.TbProcess.setStyleSheet("background-color : green")
        self.TbProcess.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.TbProcess.setFont(font)
        self.TbProcess.setVisible(False)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    # Fucntion to add the text to all the elemnts of the UI
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))


        self.BtnStart.setText(_translate("MainWindow", "Preview"))
        self.BtnSTS.setText(_translate("MainWindow", "Sit-to-stand"))
        self.BtnModel.setText(_translate("MainWindow", "Model application"))
        self.BtnRead.setText(_translate("MainWindow", "Read"))
        self.groupBox.setTitle(_translate("MainWindow", "Visualization"))
        self.RbRaw.setText(_translate("MainWindow", "Raw Video"))
        self.RbKey.setText(_translate("MainWindow", "Tracking"))
        self.RbTracking.setText(_translate("MainWindow", "Key points"))
        self.LbTime.setText(_translate("MainWindow", "Time (s):"))
        self.LbTimeDisplay.setText(_translate("MainWindow", "00:00:00"))
        self.TbTime.setText(_translate("MainWindow", "10"))
        self.LbName.setText(_translate("MainWindow", "Project name:"))
        self.TbName.setText(_translate("MainWindow", "Test"))
        self.GbProcess.setTitle(_translate("MainWindow", "Process video"))
        self.CbProcess.setText(_translate("MainWindow", "Enable"))

    # Initial configuration of the UI elements
    def initialUi(self):
        self.BtnModel.setVisible(True)
        self.BtnStart.setVisible(True)
        self.BtnSTS.setVisible(True)
        self.LbTime.setVisible(True)
        self.LbName.setVisible(True)
        self.TbName.setVisible(True)
        self.TbTime.setVisible(True)
        self.GbProcess.setVisible(True)
        self.groupBox.setVisible(True)
    
    # Turnoff all the UI elements
    def allOffUi(self):
        self.BtnModel.setVisible(False)
        self.BtnStart.setVisible(False)
        self.BtnSTS.setVisible(False)
        self.LbTime.setVisible(False)
        self.LbName.setVisible(False)
        self.TbName.setVisible(False)
        self.TbTime.setVisible(False)
        self.LbTimeDisplay.setVisible(False)
        self.GbProcess.setVisible(False)
        self.TbHip.setVisible(False)
        self.LbHip.setVisible(False)
        self.TbKnee.setVisible(False)
        self.LbKnee.setVisible(False)
        self.TbAnkle.setVisible(False)
        self.LbAnkle.setVisible(False)
        self.PictureBoxC.setVisible(False)
        self.PictureBoxD.setVisible(False)
        self.TbProcess.setVisible(False)
    
    # Configuration of the UI elements for the recording
    def recordUi(self):
        self.LbTimeDisplay.setVisible(True)
        self.TbHip.setVisible(True)
        self.LbHip.setVisible(True)
        self.TbKnee.setVisible(True)
        self.LbKnee.setVisible(True)
        self.TbAnkle.setVisible(True)
        self.LbAnkle.setVisible(True)

    # Configuration of the UI elements for the model application
    def modelUi(self):
        self.PictureBoxC.setVisible(True)
        self.PictureBoxD.setVisible(True)

    # Configuration of the UI elements for the communication
    def commUi(self):
        self.TbProcess.setVisible(True)
        self.TbProcess.setText("Waiting for a\nconnection")
        self.TbProcess.setStyleSheet("background-color : yellow")
        self.groupBox.setVisible(False)

    # Checks if a serial communication needs to be established 
    # This function is called when the user clicks the "Start" button
    def serialCommunication(self):
        self.allOffUi()                                                         # Hides all the UI elements
        self.commUi()                                                           # Shows the communication UI elements
        msgBox = QMessageBox()                                                  # Creates a message box to ask the user if they want to enable serial communication
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(f"Do you want to enable serial communication?")
        msgBox.setWindowTitle("Serial Communication...")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        returnValue = msgBox.exec()
        if(returnValue == QMessageBox.Yes):                                     # If the user clicks "Yes", the serial communication is established
            self.commFlag = 1                                                   # Sets the communication flag to 1
            self.conn = ML.comunication()                                       # Creates a communication object
            message = "Connection established, Tei just check the line 747"                                  # Message to be sent to the serial port
            self.conn.sendall(message.encode('utf-8'))

    # Function to change the value of the videoFlag variable based on the selected radio button
    def visualizationChange(self):
        if(self.RbRaw.isChecked()):             # To display the raw video
            self.videoFlag = 0
        elif(self.RbTracking.isChecked()):      # To display the keypoints
            self.videoFlag = 1
        else:
            self.videoFlag = 2                  # To display the skeleton

    # Function to define the pipeline for the cameras
    def getPipeline(self):
        pipeline = dai.Pipeline()                                                       # Creates a pipeline object to define the camera configuration
        cam_mono = pipeline.create(dai.node.MonoCamera)                                 # Creates a mono camera node
        cam_mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)     # Sets the resolution of the camera to 400p
        cam_mono.setCamera("left")                                                      # Sets the camera to the left camera
        manip = pipeline.create(dai.node.ImageManip)                                    # It's a tool used for reshape, crop and rotate images
        manip.initialConfig.setResize(300,300)                                          # Sets the size of the image to 300x300 pixels
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)                     # It's configure to read a grayscale image

        xout_mono_crop = pipeline.create(dai.node.XLinkOut)                             # Creates an output node to send the crop image to the host
        xout_mono_crop.setStreamName("crop")                                            # Sets the name of the output stream to "crop"

        cam_mono.out.link(manip.inputImage)                                             # Links the camera output to the image manipulation node manip input
        manip.out.link(xout_mono_crop.input)                                            # Links the manip output to the output node xout_mono_crop input

        return pipeline                                                                 # Returns the pipeline object

    # Function to initialize the cameras
    def worker(self,device_info, stack, devices):
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4                                  # OpenVINO is used to optimice the model aplication of the camera, but these are not used in this project
        usb2_mode = False                                                                       # USB2 mode is used to connect the camera to the host computer
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))      # Creates a device object to connect to the camera

        print("=== Connected to " + device_info.getMxId())
        device.startPipeline(self.getPipeline())                                                # Starts the pipeline of the define camera

        devices[device.getMxId()] = {                                                           # Creates a dictionary to store the device information and the output queues
            'crop': device.getOutputQueue(name="crop"),                                         # Gets the output queue for the crop image, this will work as a stream to send the image to the host
        }

    # Configuration for the recording of new test
    # This function is called when the user clicks the "Start" button
    def newTest(self):
        # Generates a new path based on the mainPath and a new folder named as the text in the TbName
        nameAux = datetime.datetime.today()                                                                                     # Gets the current date and time from the system
        self.name = str(nameAux.date())+'-'+str(nameAux.time().hour)+'-'+ str("{:02d}".format(int(nameAux.time().minute)))      # Creates a name for the test based on the current date and time
        self.testPath = self.mainPath + self.TbName.text()                                                                      # Takes the mainpath and the proyect name from the text box to generate the path for the test
        if(not(os.path.exists(self.testPath))):                                                                                 # If the proyect path does not exist
            os.mkdir(self.testPath)                                                                                             # Create a new folder with the name of the project
        
        i=0                                                                                                                     # Initialize a counter to check if the folder already exists
        self.auxPath = self.testPath + '/'+ self.name                                                                           # It takes the project path and the name of the test to generate a new path for the test
        while(os.path.exists(self.auxPath)):                                                                                    # Keeps tring to create a new folder until it finds a name that does not exist
            i += 1                                                                                                              # Increment the counter
            self.auxPath = self.testPath + '/'+ self.name + '-' + str(i)                                                        # Change the name of the test if the name already exists
        self.name = self.name + '-' + str(i)                                                                                    # Change the name of the test if the name already exists
        os.mkdir(self.testPath + '/' + self.name)                                                                               # Create a new folder with the name of the test

        self.dataPath = ML.dataFileInitializer(self.testPath,self.name,self.fileData)                                           # Creates a new file to store the movement data of the test
        self.timePath = self.timeInitializeFile(self.testPath, self.name, self.fileTime)                                        # Creates a new file to store the time data of the test
        self.videoAPath = self.testPath + '/' + self.name + '-A.avi'                                                            # Creates a path to store the video of the camera A
        self.videoBPath = self.testPath + '/' + self.name + '-B.avi'                                                            # Creates a path to store the video of the camera B
        self.closeFlag = 0                                                                                                      # Sets the close flag to 0, to indicate that the program is not closing
        self.frameCount = 0                                                                                                     # Sets the frame counter to 0, this is used to count the number of frames taken during the test

        self.RecordA = cv2.VideoWriter(self.videoAPath, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.widthVideo,self.heightVideo), isColor=True)  # Creates a video writer object to record the video of the camera A
        self.RecordB = cv2.VideoWriter(self.videoBPath, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.widthVideo,self.heightVideo), isColor=True)  # Creates a video writer object to record the video of the camera B

    # Function to manage the recording of new test
    # This function is called when the user clicks the "Preview", "Start" or "Stop" button
    def StartRecord(self):
        if(self.BtnStart.text() == 'Preview'):                                                                  # If the button text is "Preview"
            self.serialCommunication()                                                                          # Check if a serial communication with a C++ program needs to be established, in the case the comunication is needed the program will wait for a connection
            self.allOffUi()                                                                                     # Hides all the UI elements
            self.initialUi()                                                                                    # Shows the initial UI elements
            with contextlib.ExitStack() as stack:                                                               # Creates a stack to manage the resources used by the cameras
                device_infos = dai.Device.getAllAvailableDevices()                                              # Gets the list of all available devices, this are the camera id numbers
                if len(device_infos) == 0:                                                                      # If there are no devices available
                    QMessageBox.information(self,"Device search...",                                            # Popup message box to inform the user that there are no devices available
                      "There are no devices detected")
                else:                                                                                           # If there are devices available
                    self.BtnStart.setText('Start')                                                              # Change the button text to "Start"
                    print("Found", len(device_infos), "devices")                                                # Print the number of devices found, this can be omitted
                    devices = {}                                                                                # Creates a dictionary to store the device information and the output queues
                    threads = []                                                                                # Creates a list to store the threads used to connect to the devices

                    for device_info in device_infos:                                                            # Checks each device in the list of available devices
                        thread = threading.Thread(target=self.worker, args=(device_info, stack, devices))       # Creates a thread to connect to the device
                        thread.start()                                                                          # Initializes the conection to the device                  
                        threads.append(thread)                                                                  # Adds the thread to the list of threads

                    for t in threads:                                                                           # Waits for all threads to finish
                        t.join()                                                                                # When the camera is connected, the thread will finish and the program will continue

                    while(not self.closeFlag):                                                                  # Generates a continuous loop to grab new frames from the cameras
                        for mxid, q in devices.items():                                                         # Checks the output queues of each device
                            if(mxid == '18443010E1D26D0E00'):                                                   # If the device id is 18443010E1D26D0E00, this is the camera A
                                f = 0                                                                           # The flag gets a value of 0 for camera A
                            else:
                                f = 1                                                                           # The flag gets a value of 1 for camera B
                            in_crop = q['crop'].tryGet()                                                        # Gets the output queue 'crop' from the device, this is the image manipulation node output so is going to be a 300x300 greyscale image
                            if in_crop is not None:                                                             # If the output queue is not empty
                                self.frame[f] = np.copy(in_crop.getCvFrame())                                   # If the packet from camera is present, the frame is retrive in OpenCV format using getCvFrame

                                if(self.timerFlag):                                                             # If the timer flag is set to 1, this means that the recording is in progress
                                    if (self.initialtime == 0):                                                 # If the initial time is 0, this means that the recording is not started yet
                                        self.initialtime = self.dt.timestamp()                                  # Sets the initial time to the current time of the system
                                        self.timePerod = float(self.TbTime.text())                              # Reads the time period from the text box and converts it to float
                                        self.goaltime = self.initialtime + self.timePerod                       # Sets the goal time to the initial time plus the time period
                                    if(mxid == '18443010E1D26D0E00'):                                           # If the device id is 18443010E1D26D0E00, this is the camera A
                                        self.WriteTime(self.timePath)                                           # Writes the time data to the file
                                        self.frameCount += 1                                                    # Increments the frame counter by one                   
                                        self.RecordA.write(self.frame[f])                                       # Writes the frame to the video file of camera A
                                    else:
                                        self.RecordB.write(self.frame[f])                                       # Writes the frame to the video file of camera B
                            if(self.CbProcess.isChecked()):                                                     # If the process checkbox is checked, this means that the image processing is enabled in the live video 
                                self.snapFlag = 1                                                               # Sets the snap flag to 1, this means that the image processing is enabled

                        if(cv2.waitKey(1) == ord('q')):                                                         # This is used to manage the infinite loop generated to aquire the frames from the cameras, and the 'q' key is used to break but this is not commonly used
                            break
        elif(self.BtnStart.text() == 'Start'):                                                                  # If the button text is "Start"
            self.allOffUi()                                                                                     # Hides all the UI elements
            self.recordUi()                                                                                     # Shows the recording UI elements
            self.newTest()                                                                                      # Initializes the cameras and the video writer objects to record the video  
            self.timerFlag = 1                                                                                  # Sets the timer flag to 1, this means that the recording is going to start
            self.initialtime = 0                                                                                # Sets the initial time to 0, this means that the recording is not started yet
            self.BtnStart.setText("Stop")                                                                       # Change the button text to "Stop"                         
        else:                                                                                                   # If the button text is "Stop"
            self.snapFlag = 0                                                                                   # Sets the snap flag to 0, to stop the image processing
            self.timerFlag = 0                                                                                  # Sets the timer flag to 0, this finish the recording abruptly
            self.initialtime = 0                                                                                # Restart the initial time to 0
            self.allOffUi()                                                                                     # Hides all the UI elements                  
            self.initialUi()                                                                                    # Shows the initial UI elements
            self.BtnStart.setText("Preview")                                                                    # Change the button text to "Preview"
 
    # This is an alternative process to record the test, focused on the realization of the STS test
    # The sit to stand test (STS) consists of sitting down and standing up from a chair 5 times in a row, taking breaks of 10 seconds between each repetition
    def STSTest(self):
        self.STSFlag = 1                                                            # Sets the STS flag to 1, this change the behavior of the program while recording the test
        self.sitTime = 10                                                           # Sets the sit time to 10 seconds, this is used at the beginning of the test
        self.standTime = 10                                                         # Sets the stand time to 10 seconds, this is used at the end of the test
        self.repetitions = 5                                                        # Sets the number of repetitions to 5, this is the number of times the test is going to be repeated
        self.repCount = 0                                                           # Defines a counter to count the amount of repetitions done
        self.testTime = self.sitTime + self.standTime + (10*(self.repetitions-1))   # Calculates the total time of the test, this is used to set the goal time
        self.PictureBoxC.setVisible(True)                                           # Shows the label to display the sit or stand state

        self.TbTime.setText(str(self.testTime))                                     # Sets the time text box to the total time of the test
        self.TbName.setText('STS')                                                  # Sets the project name in the text box to "STS"
        
        self.StartRecord()                                                          # Call the StartRecord function, which initializes the cameras and the video writer objects to record the video
                                                                                    # After this the program will wait for the user to click the "Start" button to start the recording

    # Function to load the ML model and the scaler to apply the model to the live video
    def monitorVideo(self):
        self.modelName = 'STS_Test'                                                                                     # Defines the model to identify the states of the sit to stand action
        self.modelSTS = keras.models.load_model('./models/' + self.modelName + '/' +self.modelName + '.h5')             # Loads the keras model from the path defined in the model name
                                                                                                                        # The model can be a .h5 file or a .keras file
        scalerSTS = ML.DataExtract('./models/' + self.modelName + '/' + self.modelName + '-' + self.fileScaler,1)[0]    # Loads the scaler that corresponds to the model from a csv file
        self.scalerSTS = np.asarray(scalerSTS, dtype = np.float64, order ='C')                                          # Converts the scaler to a numpy array, this is used to scale the data before applying the model

        self.modelName = 'FallRiskTest'                                                                                 # Defines the model name that is used to detect fall risk
        self.modelFall = keras.models.load_model('./models/' + self.modelName + '/' +self.modelName + '.h5')            # Loads the keras model
        scalerFall = ML.DataExtract('./models/' + self.modelName + '/' + self.modelName + '-Scaler.csv',1)[0]           # Loads the scaler that corresponds to the model from a csv file
        self.scalerFall = np.asarray(scalerFall, dtype = np.float64, order ='C')

        self.allOffUi()                                                                                                 # Hides all the UI elements
        self.modelUi()                                                                                                  # Shows the model UI elements
        self.modelSTSFlag = 1                                                                                           # Sets the model flag to 1, this means that the model is going to be applied to the live video
        self.TbTime.setText('10000')                                                                                    # Sets the time text box to 10000 seconds, This is not necessary but is used since the models are applied to the live video without recording the video
        self.TbName.setText('Model')                                                                                    # Sets the project name in the text box to "Model"

        self.StartRecord()                                                                                              # Call the StartRecord function

    # Function to convert the image format from BGR to RGB and create a QPixmap object
    # This is used to display the image in the GUI, so there is no need to change any part of the code
    def imageFormat(self,originalImg):
        frame = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        newImg = QtGui.QPixmap.fromImage(image)
        return newImg

    # This function is used to update the Image and the time countdown
    def snap(self):
        self.dt = datetime.datetime.today()                                     # Gets the current date and time from the system

        if(self.frame[0].any() or self.frame[1].any()):                         # Checks if the frames from the cameras are not empty
            if(self.videoFlag == 0):                                            # If the video flag is 0, this means that the raw video is going to be displayed
                PixmapA = self.imageFormat(self.frame[0])                       # Display the raw frame from camera A
                PixmapB = self.imageFormat(self.frame[1])                       # Display the raw frame from camera B
            elif(self.videoFlag == 1):                                          # If the video flag is 1, this means that the keypoints are going to be displayed
                PixmapA = self.imageFormat(self.frameKey[0])                    # Display the image with keypoints from camera A
                PixmapB = self.imageFormat(self.frameKey[1])                    # Display the image with keypoints from camera B
            else:                                                               # If the video flag is 2, this means that the skeleton is going to be displayed
                PixmapA = self.imageFormat(self.frameYOLO[0])                   # Display the image with skeleton from camera A
                PixmapB = self.imageFormat(self.frameYOLO[1])                   # Display the image with skeleton from camera B
            self.PictureBoxA.setPixmap(PixmapA)                                 # Put the deseired image in the UI picture box A
            self.PictureBoxB.setPixmap(PixmapB)                                 # Put the deseired image in the UI picture box B
        
        if(self.timerFlag):                                                             # If the timer flag is set to 1, this means that the recording is in progress
            self.seconds = self.dt.timestamp()                                          # Sets the current time to the current time of the system
            timeRemaining = round((self.goaltime-self.seconds), 0)                      # Calculates the time left for the recording to finish
            self.LbTimeDisplay.setText(str(datetime.timedelta(seconds=timeRemaining)))  # Sets the time label to the time left for the recording to finish

            if(self.seconds > self.goaltime and self.initialtime != 0):                 # If the current time is greater than the goal time means that the recording is finished
                self.LbTimeDisplay.setText(str(datetime.timedelta(seconds=0)))          # Displays a time of 0 in the label
                self.snapFlag = 0                                                       # Disables the image processing
                self.STSFlag = 0                                                        # Disables the STS flag
                self.initialtime = 0                                                    # Resets the initial time to 0
                self.timerFlag = 0                                                      # Disables the timer flag
                QMessageBox.information(self,"Video record...",                         # Popup message box to inform the user that the recording is finished
                      "Video recording has finish!!!")
                self.BtnStart.setText("Start")                                          # Change the button text to "Start" 
                self.allOffUi()                                                         # Hides all the UI elements
                self.initialUi()                                                        # Shows the initial UI elements 
            if(self.STSFlag):                                                           # If the STS flag is set to 1, this means that the sit to stand test is going to be performed
                self.screenChange(self.seconds)                                         # Calls the screenChange function which coordinates the sit to stand test

    # Function to coordinate the sit to stand test
    # This change the text and color of the label that indicates if the user is sitting or standing
    def screenChange(self,now):
        self.currentTime = self.initialtime + self.sitTime                  # Sets the current time to the initial time plus the sit time
        flag = 0                                                            # sit   -> 0
                                                                            # stand -> 1
        
        if(now < self.currentTime):                                         # If the current time is less than the sit time, this means that the indicator is going to be green (sit)   
            flag = 0
        elif(self.repCount < (self.repetitions-1)):                         # Checks the number of repetitions done
            if(now < (self.currentTime + 3 + ((self.repCount)*10))):        # This change the color of the label to yellow (stand) for 3 seconds
                flag = 1                                                    # change the flag to 1 to indicate that the user must stand
            elif(now < (self.currentTime + 10 + ((self.repCount)*10))):     # This change the color of the label to green (sit) for 7 seconds, this is calculates based on the fact that the total repetition duration should be 10 seconds so if the standing time is 3 seconds the sitting time should be 7 seconds
                flag = 0                                                    # change the flag to 0 to indicate that the user must sit
            else:
                self.repCount += 1                                          # Increment the repetition counter by one after the repetition is finished
        elif(now > (self.currentTime + ((self.repetitions-1)*10))):         # If the current time is greater than the time needed to finish the repetitions leaves the label in yellow (stand) for the remaining time of the test
            flag = 1                                                        # change the flag to 1 to indicate that the user must stand 

        if(not(flag)):                                                      # If the flag is 0, Change the label to green and add the text "Sit"
            self.PictureBoxC.setText("Sit")
            self.PictureBoxC.setStyleSheet("background-color : green")
        else:                                                               # If the flag is 1, Change the label to yellow and add the text "Stand"
            self.PictureBoxC.setText("Stand")
            self.PictureBoxC.setStyleSheet("background-color : yellow")

    # Function to analice the current frame from both cameras
    # This is called every 0.33 seconds by a timer, and is limited by the system performance
    def timerAnalize(self):
        if(self.snapFlag):                                                              # If the snap flag is set to 1, this means that the image processing is enabled
            self.processFrame(self.frame,0.5)                                           # Send the current frames to be processed, the second parameter is the threshold to detect the individuals in the image

            angHip = ML.angle(self.keypoints3D,[6,12])                                  # Calculate the angle of the hip joint
            angKnee = ML.angle(self.keypoints3D,[12,14])                                # Calculate the angle of the knee joint
            angAnkle = ML.angle(self.keypoints3D,[14,16])                               # Calculate the angle of the ankle joint

            self.TbHip.setText(str(angHip))                                             # Displays the angle of the hip in a text box
            self.TbKnee.setText(str(angKnee))                                           # Displays the angle of the knee in a text box
            self.TbAnkle.setText(str(angAnkle))                                         # Displays the angle of the ankle in a text box

            if(self.commFlag):                                                          # Checks if the serial communication was stablished
                angleString = str(angHip) + ',' + str(angKnee) + ',' + str(angAnkle)    # Generates a string with the information of all the angles divided by a comma
                self.conn.sendall(angleString.encode('utf-8'))                          # This is the sender line, you just change the message to be sent and encode it to utf-8

            if(self.videoFlag == 1):                                                    # Check if the keypoints are going to be display
                frame = np.copy(self.frame)                                             # Gets a copy of the actual frame
                keypoints = np.copy(self.keypoints)                                     # Gets a copy of the keypoint information of each frame
                for f in range(len(keypoints)):                                         # Checks each of the keypoints arrays, it should be always two one for each camera
                    for point in keypoints[f]:                                          # Checks each of the points in the array
                        if(np.max(point) > 0):                                          # Validates that the point was detected
                            cv2.ellipse(frame[f], (int(point[0]),int(point[1])), (4,4),0,0,360,(255,0,0),4)     # Draws a circle in frame in the keypoint position
                    self.frameKey[f] =np.copy(frame[f])                                                         # Copies the frame with the keypoints in to a new array
    
    # Function to apply all the processing to the frames of both cameras
    def processFrame(self,frame,threshold):
        frameYOLO,keypoints,frame_height,Cam_origin = ML.skeletonDetection(frame,threshold,self.model)      # Uses the YOLO library to identify the keypoints, as a result is obtain an image with the skeleton drawn, the keypoint array, the height of the bounding box and the new origin
        self.frameYOLO = np.copy(frameYOLO)                                                                 # Copies the image with the skeleton drawn
        self.keypoints = keypoints                                                                          # Makes the keypoints arrays to a global variable
        self.keypoints3D = ML.Pose3D(keypoints,frame_height,Cam_origin)                                     # Calculates the tridimentional position of the keypoint usin de information of the two frames 

        if(self.timerFlag):                                                                                 # Check is the test has started
            self.WriteData_array(self.keypoints3D)                                                          # Writes the 3d keypoints array in a document, this is only used to plot the points using the 'Test_plot_3d_data.py' but it's not mandatory
            self.WriteRows(self.keypoints3D, self.dataPath)                                                 # Writes the 3d keypoints in the Movement file in a row format
        
        if(self.modelSTSFlag):                                                                              # Checks if it's needed to apply the ML models to the live video
            self.applyModel(self.keypoints3D)                                                               # Apply the models to the current frame
    
    # Event activated when the user closes the window
    def closeEvent(self,event):
        result = QMessageBox.question(self,                 # Open the message box
                      "Confirm Exit...",
                      "Are you sure you want to exit ?",
                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()                                      # Ignore the event to prevent closing the window, this is to avoid closing the window without confirmation

        if result == QMessageBox.Yes:
            self.closeFlag = 1                              # Set the close flag to 1 to indicate that the window should be closed
            event.accept()                                  # Accept the event to close the window

    # Writes the data of the 3d keypoints as an array
    # this is only used for ploting with the 'Test_plot_3d_data.py'
    def WriteData_array(self,keypoint3D):
        with open(self.fileArray,'w', newline='',encoding='utf-8-sig') as csv_file:     # Creates a csv file called 'Lastpose'
            Tags = ["X","Y","Z"]                                                        # Defines the headers of the file
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)                       # Generates the file writer
            csv_writer.writeheader                                                      # Defines the headers for the file
            info = {}                                                                   # Initialize an array for the data that is going to be writen
            for i in range(len(keypoint3D)):                                            # Checks each of the keypoints in the array
                info[Tags[0]]=keypoint3D[i][0]                                          # Assigns the value of the x axis
                info[Tags[1]]=keypoint3D[i][1]                                          # Assings the value of the y axis
                info[Tags[2]]=keypoint3D[i][2]                                          # Assings the value of the z axis
                csv_writer.writerow(info)                                               # Write the data in the file

    # Writes the data of the 3d keypoints as rows
    def WriteRows(self,keypoint3D, dataPath):
        with open(dataPath,'a',newline='',encoding='utf-8-sig') as csv_file:            # Creates a csv file with the termination '-Movement'
            Tags = ["Time","Hour"]                                                      # Initialize a header list starting with two, one for the values in the elapse time of the test and other with the time of the system  at the time of the frame aquisition 
            info = {}                                                                   # Initialize an array for the data that is going to be writen
            info["Time"] = self.frameCount*0.033                                        # Adds to the array the elapse time
            info["Hour"] = self.dt.time()                                               # Adds to the array the current system time
            for i in range(len(keypoint3D)):                                            # Checks each of the keypoints
                Tag = [f"X{i}",f"Y{i}",f"Z{i}"]                                         # Generates the header for each of the points, as X0,Y0,Z0,X1,Y1,Z1,....X16,Y16,Z16
                Tags += Tag                                                             # Adds the headers to a list
                info[Tag[0]]=keypoint3D[i][0]                                           # Assigns the value of the x axis
                info[Tag[1]]=keypoint3D[i][1]                                           # Assigns the value of the y axis
                info[Tag[2]]=keypoint3D[i][2]                                           # Assigns the value of the z axis
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)                       # Generates the file writer
            csv_writer.writerow(info)                                                   # Write the data in the file in a row format
    
    # Writes a file for the time data
    # This is used for the post processing to maintain the time values for the postprocessing files
    def timeInitializeFile(self, mainPath, name, termination):                  
        timePath = mainPath + '/' + name + '-' + termination                        # Generates the path to save the time values
        with open(timePath,'w',newline='',encoding='utf-8-sig') as csv_file:        # Creates the file with termination '-Time'
            Tags = ["Time","Hour"]                                                  # Initialize a header list starting with two, one for the values in the elapse time of the test and other with the time of the system  at the time of the frame aquisition 
            info = {}                                                               # Initialize an array for the data that is going to be writen
            info["Time"]="Time"                                                     # Adds the header for time
            info["Hour"]="Hour"                                                     # Adds the header for the system time of aquisition
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)                   # Generates the file writer
            csv_writer.writeheader                                                  # Defines the headers for the file
            csv_writer.writerow(info)                                               # Write the data in the file in a row format
        return timePath                                                             # Returs the path of the time file
    
    # Add new values to the Time file
    def WriteTime(self,timePath):
        with open(timePath,'a',newline='',encoding='utf-8-sig') as csv_file:    # Appends information to the file
            Tags = ["Time","Hour"]                                              # Initialize a header list starting with two, one for the values in the elapse time of the test and other with the time of the system  at the time of the frame aquisition 
            info = {}                                                           # Initialize an array for the data that is going to be writen
            info[Tags[1]]= str(self.dt.time())                                  # Adds the system time
            info[Tags[0]]= (self.frameCount)*0.033                              # Adds the elapse time of the test
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)               # Generates the file writer
            csv_writer.writerow(info)                                           # Write the data in the file in a row format
    
    # Add the Saving file for the angles
    def angleInitializer(self, mainPath,name):
        self.anglePath = mainPath + '/' + name + '-Angle.csv'
        with open(self.anglePath,'w',newline='',encoding='utf-8-sig') as csv_file:
            Tags = ["Time","Hour","Hip","Knee","Ankle"]
            info = {}
            info["Time"]="Time"
            info["Hour"]="Hour"
            info["Hip"]="Hip"
            info["Knee"]="Knee"
            info["Time"]="Time"
            info["Ankle"]="Ankle"
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
            csv_writer.writeheader
            csv_writer.writerow(info)
        return 0

    # Applies the ML models to the current frames
    # This is only used if the 'Model application' button is used
    def applyModel(self,keypoints):
        keypointsRow = self.scalerApplication(keypoints,self.scalerSTS)         # Takes the current 3d keypoints and scale them using the '-Scaler' file of the STS model
        keypointsRowFall = self.scalerApplication(keypoints,self.scalerFall)    # Takes the current 3d keypoints and scale them using the '-Scaler' file of the FallRisk model

        self.timeFeatures.append(keypointsRow)                                  # Adds the current scaled 3d keypoints of STS to an array, this is used to compile the changes in the pose in a window of time
        self.timeFeaturesFall.append(keypointsRowFall)                          # Adds the current scaled 3d keypoints of FallRisk to an array

        if(len(self.timeFeatures) == 5):                                        # Checks if the array has five simples of the pose
            sample = np.reshape(self.timeFeatures,(1,5,36))                     # It ensures that the data is in the proper shape for applying the model
            res = self.modelSTS.predict(sample, batch_size=1, verbose=0)
            #print(f'Res: {res}')
            rounded_predictions = np.argmax(res, axis=-1)
            print(f"State:{rounded_predictions}")
            if(rounded_predictions == 1):
                self.PictureBoxC.setText("Preparing to stand")
                self.PictureBoxC.setStyleSheet("background-color : yellow") 
            elif(rounded_predictions == 2):
                self.PictureBoxC.setText("Standing")
                self.PictureBoxC.setStyleSheet("background-color : orange") 
            elif(rounded_predictions == 3):
                self.PictureBoxC.setText("Stand")
                self.PictureBoxC.setStyleSheet("background-color : blue") 
            elif(rounded_predictions == 4):
                self.PictureBoxC.setText("Sitting")
                self.PictureBoxC.setStyleSheet("background-color : gray") 
            else:
                self.PictureBoxC.setText("Sit")
                self.PictureBoxC.setStyleSheet("background-color : green")

            if(rounded_predictions == 1 or rounded_predictions == 2):
                sampleFall = np.reshape(self.timeFeaturesFall,(1,5,36))
                res = self.modelFall.predict(sampleFall, batch_size=1, verbose=0)
                rounded_predictions = np.argmax(res, axis=-1)
                print(f"Pose:{rounded_predictions}")
                if(rounded_predictions == 1):
                    self.PictureBoxD.setText("OverExtend")
                    self.PictureBoxD.setStyleSheet("background-color : yellow") 
                elif(rounded_predictions == 2):
                    self.PictureBoxD.setText("Tild")
                    self.PictureBoxD.setStyleSheet("background-color : orange") 
                elif(rounded_predictions == 3):
                    self.PictureBoxD.setText("Support")
                    self.PictureBoxD.setStyleSheet("background-color : blue") 
                else:
                    self.PictureBoxD.setText("Normal")
                    self.PictureBoxD.setStyleSheet("background-color : green") 
            else:
                self.PictureBoxD.setText("")
                self.PictureBoxD.setStyleSheet("background-color : white") 
            ###
            self.timeFeatures.pop(0)
            self.timeFeaturesFall.pop(0)

    # Function to apply the scaler to the keypoints
    # It takes the keypoints and the model scaler as input and returns the scaled keypoints as a 1D array
    def scalerApplication(self,keypoints,modelScaler):
        keypoints3d = keypoints[5:]                                         # Avoid the first 5 keypoints since these correspond to the head
        keypointsRow = ML.poseRow(keypoints3d)                              # Change the keypoints to a row format
       
        scaler = MinMaxScaler(feature_range=(0,1))                          # Scaler object
        zeroRow = np.zeros(36)                                              # Create a zero row to be used as a reference for the scaler
        newdata = np.array([modelScaler,zeroRow,keypointsRow])              # Create a new data array with the scaler, zero row and keypoints row

        keypointsScaled = scaler.fit_transform(newdata.reshape(-1,36))      # Applies the scaler to the new data array
        keypointsRow = np.array(keypointsScaled[2])                         # Get the second row of the scaled data array which corresponds to the scaled keypoints row
        keypointsRow = keypointsRow.reshape(1,36)                           # Reshape the keypoints row to a 1D array

        return keypointsRow                                                 # Return the scaled keypoints row




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
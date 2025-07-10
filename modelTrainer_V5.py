from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import cv2
import csv
import numpy as np
import os
import ML_V5 as ML
from ultralytics import YOLO


class Ui_Form(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Ui_Form, self).__init__(*args, **kwargs)
        self.initialVariables()
        # YOLO parameters
        modelSize = 1       #(1-5)
        YOLO_models = {1:'yolov8n-pose.pt',2:'yolov8s-pose.pt',3:'yolov8m-pose.pt',4:'yolov8l-pose.pt',5:'yolov8x-pose.pt'}
        model_path = YOLO_models[modelSize]
        self.model = YOLO(model_path)
        self.setupUi()

    def initialVariables(self):
        self.fileLabels = 'Labels.csv'
        self.fileData = 'MovementR.csv'
        self.fileTime = 'Time.csv'
        self.fileAngle = 'Angle.csv'
        self.flagStop = 0
        self.mainPath = ''
        self.videoList = []
        self.keypointsRow = []
        self.anglesRow = []
        self.time = []
        blank = np.zeros((300,300,3))
        self.blank = blank.astype('uint8') 
        self.frame = [self.blank,self.blank]
        self.frameKey = np.copy(self.frame)
        self.frameYOLO = np.copy(self.frame)
        blankPoints = np.zeros((17,2))
        self.blankPoints = blankPoints.astype('float32') 
        self.keypoints = [self.blankPoints,self.blankPoints]
        self.frame_height = [[],[]]
        self.Cam_origin = [[],[]]
        self.count = 0
        self.wSize = 5
        self.videoLabels = np.array([])
        self.videoData = []
        self.frames = 100
        self.padding = 10
        self.framesCount = 0
        self.modelName = 'newModel'
        self.modelPath = './models/'
        self.CaptureClick = False

    def setupUi(self):
        self.setObjectName("Form")
        self.resize(1260, 530)

        self.PbA = QtWidgets.QLabel(self)
        self.PbA.setGeometry(QtCore.QRect(30, 13, 500, 500))
        self.PbA.setFrameShape(QtWidgets.QFrame.Box)
        self.PbA.setFrameShadow(QtWidgets.QFrame.Plain)
        self.PbA.setScaledContents(True)
        self.PbA.setWordWrap(False)
        self.PbA.setLineWidth(2)
        self.PbA.setObjectName("PbA")

        self.PbB = QtWidgets.QLabel(self)
        self.PbB.setGeometry(QtCore.QRect(532, 13, 500, 500))
        self.PbB.setFrameShape(QtWidgets.QFrame.Box)
        self.PbB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.PbB.setScaledContents(True)
        self.PbB.setWordWrap(False)
        self.PbB.setLineWidth(2)
        self.PbB.setObjectName("PbB")

        self.GbLabels = QtWidgets.QGroupBox(self)
        self.GbLabels.setGeometry(QtCore.QRect(1040, 100, 200, 210))
        self.GbLabels.setObjectName("GbLabels")
        self.GbLabels.setVisible(False)

        self.RbOne = QtWidgets.QRadioButton(self.GbLabels)
        self.RbOne.setGeometry(QtCore.QRect(10, 30, 180, 25))
        self.RbOne.setObjectName("RbOne")
        self.RbOne.setChecked(True)

        self.RbTwo = QtWidgets.QRadioButton(self.GbLabels)
        self.RbTwo.setGeometry(QtCore.QRect(10, 60, 180, 25))
        self.RbTwo.setObjectName("RbTwo")

        self.RbThree = QtWidgets.QRadioButton(self.GbLabels)
        self.RbThree.setGeometry(QtCore.QRect(10, 90, 180, 25))
        self.RbThree.setObjectName("RbThree")

        self.RbFour = QtWidgets.QRadioButton(self.GbLabels)
        self.RbFour.setGeometry(QtCore.QRect(10, 120, 180, 25))
        self.RbFour.setObjectName("RbFour")

        self.RbFive = QtWidgets.QRadioButton(self.GbLabels)
        self.RbFive.setGeometry(QtCore.QRect(10, 150, 180, 25))
        self.RbFive.setObjectName("RbFive")

        self.RbSix = QtWidgets.QRadioButton(self.GbLabels)
        self.RbSix.setGeometry(QtCore.QRect(10, 180, 180, 25))
        self.RbSix.setObjectName("RbSix")

        self.TbOne = QtWidgets.QLineEdit(self.GbLabels)
        self.TbOne.setGeometry(QtCore.QRect(10, 30, 180, 25))
        self.TbOne.setObjectName("TbOne")

        self.TbTwo = QtWidgets.QLineEdit(self.GbLabels)
        self.TbTwo.setGeometry(QtCore.QRect(10, 60, 180, 25))
        self.TbTwo.setObjectName("TbTwo")

        self.TbThree = QtWidgets.QLineEdit(self.GbLabels)
        self.TbThree.setGeometry(QtCore.QRect(10, 90, 180, 25))
        self.TbThree.setObjectName("TbThree")

        self.TbFour = QtWidgets.QLineEdit(self.GbLabels)
        self.TbFour.setGeometry(QtCore.QRect(10, 120, 180, 25))
        self.TbFour.setObjectName("TbFour")

        self.TbFive = QtWidgets.QLineEdit(self.GbLabels)
        self.TbFive.setGeometry(QtCore.QRect(10, 150, 180, 25))
        self.TbFive.setObjectName("TbFive")

        self.TbSix = QtWidgets.QLineEdit(self.GbLabels)
        self.TbSix.setGeometry(QtCore.QRect(10, 180, 180, 25))
        self.TbSix.setObjectName("TbSix")

        self.TbName = QtWidgets.QLineEdit(self)
        self.TbName.setGeometry(QtCore.QRect(1040, 13, 200, 30))
        self.TbName.setObjectName("TbName")
        self.TbName.setEnabled(False)

        self.PbLoad = QtWidgets.QPushButton(self)
        self.PbLoad.setGeometry(QtCore.QRect(1040, 53, 200, 41))
        self.PbLoad.setObjectName("PbLoad")
        self.PbLoad.clicked.connect(self.trainingFile)

        self.PbLoadVideo = QtWidgets.QPushButton(self)
        self.PbLoadVideo.setGeometry(QtCore.QRect(1040, 104, 200, 41))
        self.PbLoadVideo.setObjectName("PbLoadVideo")
        self.PbLoadVideo.clicked.connect(self.singleFile)

        self.PbNext = QtWidgets.QPushButton(self)
        self.PbNext.setGeometry(QtCore.QRect(1040, 320, 200, 41))
        self.PbNext.setObjectName("PbNext")
        self.PbNext.clicked.connect(self.setLabels)
        self.PbNext.setVisible(False)

        self.PictureBoxH = QtWidgets.QLabel(self)
        self.PictureBoxH.setGeometry(QtCore.QRect(1040, 385, 200, 41))
        self.PictureBoxH.setFrameShape(QtWidgets.QFrame.Panel)
        self.PictureBoxH.setLineWidth(1)
        self.PictureBoxH.setAutoFillBackground(True)
        self.PictureBoxH.setObjectName("PictureBoxH")
        self.PictureBoxH.setVisible(False)

        self.TbLoading = QtWidgets.QLabel(self)
        self.TbLoading.setGeometry(QtCore.QRect(1050, 353, 200, 41))
        self.TbLoading.setText("Loading...")
        self.TbLoading.setObjectName("TbLoading")
        self.TbLoading.setVisible(False)

        self.PictureBoxL = QtWidgets.QLabel(self)
        self.PictureBoxL.setGeometry(QtCore.QRect(1045, 390, 190, 31))
        self.PictureBoxL.setFrameShape(QtWidgets.QFrame.Panel)
        self.PictureBoxL.setLineWidth(2)
        self.PictureBoxL.setAutoFillBackground(True)
        self.PictureBoxL.setObjectName("PictureBoxL")
        self.PictureBoxL.setVisible(False)

        self.PictureBoxLB = QtWidgets.QLabel(self)
        self.PictureBoxLB.setGeometry(QtCore.QRect(1045, 390, 0, 31))
        self.PictureBoxLB.setFrameShape(QtWidgets.QFrame.Panel)
        self.PictureBoxLB.setLineWidth(2)
        self.PictureBoxLB.setAutoFillBackground(True)
        self.PictureBoxLB.setObjectName("PictureBoxLB")
        self.PictureBoxLB.setStyleSheet("background-color : green")
        self.PictureBoxLB.setVisible(False)

        self.TbFrames = QtWidgets.QLabel(self)
        self.TbFrames.setGeometry(QtCore.QRect(1040, 390, 200, 31))
        self.TbFrames.setText("0/0")
        self.TbFrames.setObjectName("TbFrames")
        self.TbFrames.setAlignment(QtCore.Qt.AlignCenter)
        self.TbFrames.setVisible(False)

        self.GbProcess = QtWidgets.QGroupBox(self)
        self.GbProcess.setGeometry(QtCore.QRect(1040, 150, 200, 130))
        self.GbProcess.setObjectName("GbProcess")

        self.CbOne = QtWidgets.QCheckBox(self.GbProcess)
        self.CbOne.setGeometry(QtCore.QRect(10, 30, 150, 25))
        self.CbOne.setObjectName("CbOne")
        self.CbOne.setChecked(True)

        self.CbTwo = QtWidgets.QCheckBox(self.GbProcess)
        self.CbTwo.setGeometry(QtCore.QRect(10, 60, 150, 25))
        self.CbTwo.setObjectName("CbTwo")

        self.CbThree = QtWidgets.QCheckBox(self.GbProcess)
        self.CbThree.setGeometry(QtCore.QRect(10, 90, 150, 25))
        self.CbThree.setObjectName("CbThree")

        self.TbProcess = QtWidgets.QLabel(self)
        self.TbProcess.setGeometry(QtCore.QRect(1040, 53, 200, 41))
        self.TbProcess.setFrameShape(QtWidgets.QFrame.Panel)
        self.TbProcess.setLineWidth(2)
        self.TbProcess.setAutoFillBackground(True)
        self.TbProcess.setText("Labeling")
        self.TbProcess.setObjectName("TbProcess")
        self.TbProcess.setStyleSheet("background-color : green")
        self.TbProcess.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(13)
        self.TbProcess.setFont(font)
        self.TbProcess.setVisible(False)
        
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))

        self.PbA.setText(_translate("Form", ""))
        self.PbB.setText(_translate("Form", ""))
        self.GbLabels.setTitle(_translate("Form", "Labels"))
        self.RbOne.setText(_translate("Form", "0"))
        self.RbTwo.setText(_translate("Form", "1"))
        self.RbThree.setText(_translate("Form", "2"))
        self.RbFour.setText(_translate("Form", "3"))
        self.RbFive.setText(_translate("Form", "4"))
        self.RbSix.setText(_translate("Form", "5"))
        self.TbOne.setText(_translate("Form", ""))
        self.TbTwo.setText(_translate("Form", ""))
        self.TbThree.setText(_translate("Form", ""))
        self.TbFour.setText(_translate("Form", ""))
        self.TbFive.setText(_translate("Form", ""))
        self.TbSix.setText(_translate("Form", ""))
        self.GbProcess.setTitle(_translate("Form", "Process"))
        self.CbOne.setText(_translate("Form", "Labeling"))
        self.CbTwo.setText(_translate("Form", "Pose detection"))
        self.CbThree.setText(_translate("Form", "Model training"))
        self.PbLoad.setText(_translate("Form", "Load Training file"))
        self.PbLoadVideo.setText(_translate("Form", "Load single video"))
        self.PbNext.setText(_translate("Form", "Set Labels"))
        self.TbName.setText(_translate("Form", f"{self.modelName}"))

    def labelingUi(self):
        self.GbLabels.setVisible(True)
        self.PbNext.setVisible(True)
        self.PictureBoxH.setVisible(True)
        self.PictureBoxL.setVisible(True)
        self.PictureBoxLB.setVisible(True)
        self.TbFrames.setVisible(True)
        self.TbLoading.setVisible(True)
        self.TbProcess.setVisible(True)
        self.TbProcess.setText("Labeling")
        self.TbProcess.setStyleSheet("background-color : green")
        self.TbOne.setVisible(True)
        self.TbTwo.setVisible(True)
        self.TbThree.setVisible(True)
        self.TbFour.setVisible(True)
        self.TbFive.setVisible(True)
        self.TbSix.setVisible(True)
    
    def processgUi(self):
        self.PictureBoxH.setVisible(True)
        self.PictureBoxL.setVisible(True)
        self.PictureBoxLB.setVisible(True)
        self.TbFrames.setVisible(True)
        self.TbLoading.setVisible(True)
        self.TbProcess.setVisible(True)
        self.TbProcess.setText("Processing")
        self.TbProcess.setStyleSheet("background-color : yellow")

    def trainingUi(self):
        self.TbProcess.setVisible(True)
        self.TbProcess.setText('Training')
        self.TbProcess.setStyleSheet("background-color : red")

    def initialUi(self):
        self.GbProcess.setVisible(True)
        self.PbLoad.setVisible(True)
        self.PbLoadVideo.setVisible(True)

    def allOffUi(self):
        self.PbLoad.setVisible(False)
        self.GbLabels.setVisible(False)
        self.PbNext.setVisible(False)
        self.PictureBoxH.setVisible(False)
        self.PictureBoxL.setVisible(False)
        self.PictureBoxLB.setVisible(False)
        self.TbFrames.setVisible(False)
        self.TbLoading.setVisible(False)
        self.GbProcess.setVisible(False)
        self.PbLoadVideo.setVisible(False)
        self.TbProcess.setVisible(False)

    def trainingFile(self):
        dialogResult = QtWidgets.QFileDialog.getOpenFileName(self,'List of videos...','','*.csv')
        documentPath = dialogResult[0]
        documentPath = documentPath.split('/')
        self.modelName = documentPath.pop((len(documentPath)-1))
        self.modelName = self.modelName.split('.')[0]
        self.TbName.setText(self.modelName)
        mainPath = ''
        for dic in documentPath:
            mainPath += dic + '/'
        videoList = ML.DataExtract(dialogResult[0],0)
        self.mainPath = mainPath
        self.videoList = videoList

        self.modelPath += self.modelName + '/'

        self.readVideo()

    def singleFile(self):
        dialogResult = QtWidgets.QFileDialog.getExistingDirectory(self,'Video selection...','')
        documentPath = dialogResult
        documentPath = documentPath.split('/')
        videoName = documentPath.pop((len(documentPath)-1))
        self.TbName.setText(videoName)
        mainPath = ''
        for dic in documentPath:
            mainPath += dic + '/'
        self.videoList = []
        self.mainPath = mainPath
        self.videoList.append(videoName)

        if(not(self.CbThree.isChecked())):
            self.readVideo()
        else:
            QtWidgets.QMessageBox.information(self,"Training error...",
            "You can't train an algorithm with only one video")

    def readVideo(self):

        if(self.CbOne.isChecked() or self.CbThree.isChecked()):
            self.flagLabels = self.CheckLabels()
        
        if(self.CbTwo.isChecked() or self.CbThree.isChecked()):
            self.flagData = self.CheckMovement()

        if(self.CbOne.isChecked() or self.CbThree.isChecked()):
            self.allOffUi()
            self.labelingUi()
            self.videoLabeling()
        elif(self.CbTwo.isChecked() or self.CbThree.isChecked()):
            self.allOffUi()
            self.processgUi()
            self.videoProcessing()

    def CheckLabels(self):
        flagLabel = 0
        for video in self.videoList:
            readVideoPath = self.mainPath + video
            labelPath = readVideoPath + '/' + video + '-' + self.fileLabels
            if(os.path.exists(labelPath)):
                flagLabel = 1

        if(flagLabel):
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Information)
            msgBox.setText(f"Some videos are already labeled, Do you want to use the same labels?")
            msgBox.setWindowTitle("Label videos...")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            returnValue = msgBox.exec()
            if(returnValue == QtWidgets.QMessageBox.No):
                flagLabel = 0
        
        return flagLabel
    
    def CheckMovement(self):
        flagData = 0
        for video in self.videoList:
            readVideoPath = self.mainPath + video
            dataPath = readVideoPath + '/' + video + '-' + self.fileData
            if(os.path.exists(dataPath)):
                flagData = 1

        if(flagData):
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Information)
            msgBox.setText(f"Some videos are already process, Do you want to use the same data?")
            msgBox.setWindowTitle("Label videos...")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            returnValue = msgBox.exec()
            if(returnValue == QtWidgets.QMessageBox.No):
                flagData = 0
        
        return flagData
        
    def videoLabeling(self):
        if(self.count < len(self.videoList)):
            self.video = self.videoList[self.count]
            readVideoPath = self.mainPath + self.video

            if(os.path.exists(readVideoPath)):
                labelPath = readVideoPath + '/' + self.video + '-' + self.fileLabels
                if(os.path.exists(labelPath) and self.flagLabels):
                    self.addLabels(labelPath)
                    self.count += 1
                    self.videoLabeling()
                else:
                    self.readMainPath = readVideoPath
                    self.readName = self.video
                    self.localLabelPath = self.localFileInititalizer(self.readMainPath,self.readName)
                    videoAPath = self.readMainPath + '/' + self.readName + '-A.avi'
                    videoBPath = self.readMainPath + '/' + self.readName + '-B.avi'
                    if(os.path.exists(videoAPath)):
                        print('True')
                    self.recordedCaptureA = cv2.VideoCapture(videoAPath)
                    self.recordedCaptureB = cv2.VideoCapture(videoBPath)
                    self.frames = self.recordedCaptureA.get(cv2.CAP_PROP_FRAME_COUNT) 
                    self.readFrame()
                
            else:
                QtWidgets.QMessageBox.information(self,"Omitted videos...",
                f"The video {self.video} have been omitted.")
                self.count += 1
                self.videoLabeling()
        else:
            # self.PbNext.setEnabled(False)
            # self.GbLabels.setEnabled(False)
            # self.PbLoad.setEnabled(True)
            #self.TbName.setEnabled(True)
            QtWidgets.QMessageBox.information(self,"Finish videos...",
            f"All the videos have been label!!!")
            self.count = 0
            if(self.CbTwo.isChecked() or self.CbThree.isChecked()):
                self.allOffUi()
                self.processgUi()
                self.videoProcessing()
            else:
                self.allOffUi()
                self.initialVariables()
                self.initialUi()
    
    def videoProcessing(self):
        if(self.count < len(self.videoList)):
            self.video = self.videoList[self.count]
            readVideoPath = self.mainPath + self.video

            if(os.path.exists(readVideoPath)):
                dataPath = readVideoPath + '/' + self.video + '-' + self.fileData
                if(os.path.exists(dataPath) and self.flagData):
                    self.addData(dataPath)
                    self.count += 1
                    self.videoProcessing()
                else:
                    self.readMainPath = readVideoPath
                    self.readName = self.video
                    self.dataPath = dataPath
                    self.localDataPath = ML.dataFileInitializer(self.readMainPath,self.readName,self.fileData)
                    self.angleInitializer(self.readMainPath,self.readName)
                    timePath = readVideoPath + '/' + self.video + '-' + self.fileTime
                    if(os.path.exists(timePath)):
                        self.timeData = (ML.DataExtract(timePath,1))[1:]
                    else:
                        self.timeData = []
                    videoAPath = self.readMainPath + '/' + self.readName + '-A.avi'
                    videoBPath = self.readMainPath + '/' + self.readName + '-B.avi'
                    self.recordedCaptureA = cv2.VideoCapture(videoAPath)
                    self.recordedCaptureB = cv2.VideoCapture(videoBPath)
                    self.frames = self.recordedCaptureA.get(cv2.CAP_PROP_FRAME_COUNT) 
                    self.fps = self.recordedCaptureA.get(cv2.CAP_PROP_FPS)

                    #### Save process video
                    self.videoAProcessPath = readVideoPath + '/' + self.readName + '-A-Process.avi'
                    self.videoBProcessPath = readVideoPath + '/' + self.readName + '-B-Process.avi'
                    self.RecordA = cv2.VideoWriter(self.videoAProcessPath, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (300,300), isColor=True)
                    self.RecordB = cv2.VideoWriter(self.videoBProcessPath, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (300,300), isColor=True)

                    retA, self.frame[0] = self.recordedCaptureA.read()
                    retB, self.frame[1] = self.recordedCaptureB.read()
                    PixmapA = self.imageFormat(self.frame[0])
                    self.PbA.setPixmap(PixmapA)
                    PixmapB = self.imageFormat(self.frame[1])
                    self.PbB.setPixmap(PixmapB)
                    self.Cam_origin = [[],[]]
                    self.referencePoint()
            else:
                QtWidgets.QMessageBox.information(self,"Omitted videos...",
                f"The video {self.video} have been omitted.")
                self.count += 1
                self.videoProcessing()
        else:
            QtWidgets.QMessageBox.information(self,"Finish videos...",
            f"All the videos have been process!!!")
            if(self.CbThree.isChecked()):
                if(not(os.path.exists(self.modelPath))):
                    # create new single directory
                    os.mkdir(self.modelPath)
                self.trainModel()
            else:
                self.allOffUi()
                self.initialVariables()
                self.initialUi()
            # except:
            #     QtWidgets.QMessageBox.information(self,"Training unsuccessful...",
            #     f"There has been a problem with the data, please make sure the labeling and processing of each video is correct")
            self.count = 0

    def videoProcessingCon(self):
        self.readAnalize()
        self.addData(self.dataPath)
        self.count +=1
        self.videoProcessing()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            if(self.CaptureClick == True):
                conv = 300/self.PbA.width()
                PbAPos = self.PbA.mapFrom(self,event.pos())
                x_A = PbAPos.x()
                y_A = PbAPos.y()
                # print(f"W: {self.PbA.width()}")
                # print(f"H: {self.PbA.height()}")
                if(x_A < int(self.PbA.width()) and x_A > 0 and y_A < int(self.PbA.height()) and y_A > 0):
                    self.Cam_origin[0] = [int(x_A*conv),int(y_A*conv)]
                    print(self.Cam_origin)

                PbBPos = self.PbB.mapFrom(self,event.pos())
                x_B = PbBPos.x()
                y_B = PbBPos.y()
                if(x_B < int(self.PbA.width()) and x_B > 0 and y_B < int(self.PbA.height()) and y_B > 0):
                    self.Cam_origin[1] = [int(x_B*conv),int(y_B*conv)]
                    print(self.Cam_origin)
                
                self.drawRefPoint()
                
                if(self.Cam_origin[0]!=[] and self.Cam_origin[1]!=[]):
                    msgBox = QtWidgets.QMessageBox()
                    msgBox.setIcon(QtWidgets.QMessageBox.Information)
                    msgBox.setText(f"Is the origin set in each display ok?")
                    msgBox.setWindowTitle("Reference point...")
                    msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    returnValue = msgBox.exec()
                    if(returnValue == QtWidgets.QMessageBox.No):
                        self.Cam_origin=[[],[]]
                    else:
                        print(self.Cam_origin)
                        self.videoProcessingCon()
                        self.CaptureClick = False


    def referencePoint(self):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.setText(f"Do you want to set a referrence point as the origin?")
        msgBox.setWindowTitle("Reference point...")
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        returnValue = msgBox.exec()
        if(returnValue == QtWidgets.QMessageBox.No):
            self.Cam_origin = [[0,300],[0,300]]
            print(self.Cam_origin)
            self.videoProcessingCon()
            return False
        self.CaptureClick = True
        QtWidgets.QMessageBox.information(self,"Reference point selection...",
            f"Use the right click to select the reference point in each of the displays.")
        
    def drawRefPoint(self):
        AuxImg = np.copy(self.frame)
        for i in range(len(self.Cam_origin)):
            if(self.Cam_origin[i] != []):
                print(self.Cam_origin[i])
                cv2.line(AuxImg[i],[0,self.Cam_origin[i][1]],[300,self.Cam_origin[i][1]],(255,0,0),2)
                cv2.line(AuxImg[i],[self.Cam_origin[i][0],0],[self.Cam_origin[i][0],300],(255,0,0),2)
                cv2.drawMarker(AuxImg[i],self.Cam_origin[i],(0,0,255),cv2.MARKER_CROSS,20,2)
                
        
        PixmapA = self.imageFormat(AuxImg[0])
        self.PbA.setPixmap(PixmapA)
        PixmapB = self.imageFormat(AuxImg[1])
        self.PbB.setPixmap(PixmapB)

    def readAnalize(self):
        retA = True
        retB = True

        threshold = 0.6
        timeFeatures = []
        videoData = []
        while (retA and retB and not(self.flagStop)):
            if(self.frames > 0):
                percent = int((self.framesCount*190)/self.frames)
                self.PictureBoxLB.setGeometry(QtCore.QRect(1045, 390, percent, 31))
            self.framesCount += 1
            self.TbFrames.setText(f"{self.framesCount}/{self.frames}")

            if(len(self.frame[0]) != 300):
                self.frame[0] = cv2.resize(self.frame[0], (300,300), interpolation=cv2.INTER_AREA)
                self.frame[1] = cv2.resize(self.frame[1], (300,300), interpolation=cv2.INTER_AREA)

            PixmapA = self.imageFormat(self.frame[0])
            self.PbA.setPixmap(PixmapA)
            PixmapB = self.imageFormat(self.frame[1])
            self.PbB.setPixmap(PixmapB)

            self.processFrame(self.frame,threshold)

            ## Knee angle calculation
            angHip = ML.angle(self.keypoints3d,[6,12])
            angKnee = ML.angle(self.keypoints3d,[12,14])
            angAnkle = ML.angle(self.keypoints3d,[14,16])
            angles = [angHip,angKnee,angAnkle]
            self.anglesRow.append(angles)
            #self.angleFrame(angles)
            #############################

        #     #########
        #     keypointsRow = keypointsRow[15:]
        #     #########

        #     timeFeatures.append(keypointsRow)
        #     #print('result:')
        #     #print(keypointsRow)
        #     if(len(timeFeatures) == 5):
        #         videoData.append(np.copy(timeFeatures))
        #         timeFeatures.pop(0)

            retA, self.frame[0] = self.recordedCaptureA.read()
            retB, self.frame[1] = self.recordedCaptureB.read()

            if (cv2.waitKey(25) & 0xFF == ord('q')):
                break
        # videoData = np.array(videoData)
        self.framesCount = 0

        print(self.video)
        print(len(videoData))

        newData = ML.intepolate(self.keypointsRow,self.time)
        newData = ML.filterEMA(newData,0.5)
        self.WriteRows2(newData,self.localDataPath)

        newAngles = ML.intepolate(self.anglesRow,self.time)
        newAngles = ML.filterEMA(newAngles,0.5)
        self.angleFrame2(newData)
    
    def processFrame(self,frame,threshold):
        frameYOLO,keypoints,frame_height,Cam_origin = ML.skeletonDetection(frame,threshold,self.model)
        Cam_origin = self.Cam_origin


        self.RecordA.write(frameYOLO[0])
        if(len(frame)>1):
            self.RecordB.write(frameYOLO[1])
                        
        self.keypoints = keypoints
        self.keypoints3d = ML.Pose3D(keypoints,frame_height,Cam_origin)
        self.keypointsRow.append(ML.poseRow(self.keypoints3d))
        self.time.append((self.framesCount-1)*0.033)

    def readFrame(self):
        retA, self.frame[0] = self.recordedCaptureA.read()
        retB, self.frame[1] = self.recordedCaptureB.read()

        if(retA and retB):
            if(self.frames > 0):
                percent = int((self.framesCount*190)/self.frames)
                self.PictureBoxLB.setGeometry(QtCore.QRect(1045, 390, percent, 31))
            self.framesCount += 1
            self.TbFrames.setText(f"{self.framesCount}/{self.frames}")

            if(len(self.frame[0]) != 300):
                self.frame[0] = cv2.resize(self.frame[0], (300,300), interpolation=cv2.INTER_AREA)
                self.frame[1] = cv2.resize(self.frame[1], (300,300), interpolation=cv2.INTER_AREA)
            
            PixmapA = self.imageFormat(self.frame[0])
            self.PbA.setPixmap(PixmapA)
            PixmapB = self.imageFormat(self.frame[1])
            self.PbB.setPixmap(PixmapB)
        else:
            QtWidgets.QMessageBox.information(self,"Video analysis...",
                f"{self.readName} video analysis has finish!!!")
            self.count += 1
            self.framesCount = 0
            self.videoLabeling()
    
    def labelFrame(self):
        with open(self.localLabelPath,'a',newline='',encoding='utf-8-sig') as csv_file:
            Tags = ["Label"]
            if(self.RbOne.isChecked()):
                info = {'Label' : 0}        #   Sit
            if(self.RbTwo.isChecked()):     
                info = {'Label' : 1}        #   Preparing to stand
            if(self.RbThree.isChecked()):   
                info = {'Label' : 2}        #   Standing
            if(self.RbFour.isChecked()):
                info = {'Label' : 3}        #   Stand
            if(self.RbFive.isChecked()):
                info = {'Label' : 4}
            if(self.RbSix.isChecked()):
                info = {'Label' : 5}   
            if(self.framesCount>((self.wSize-1)*self.padding)):
                self.videoLabels = np.append(self.videoLabels,info['Label'])
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
            csv_writer.writerow(info)
        self.readFrame()

    def localFileInititalizer(self, mainPath,name):
        labelPath = mainPath + '/' + name + '-' + self.fileLabels
        with open(labelPath,'w',newline='',encoding='utf-8-sig') as csv_file:
            Tags = ["Label"]
            info = {}
            info["Label"]="Label"
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
            csv_writer.writeheader
            csv_writer.writerow(info)
        return labelPath

    def WriteRows(self,keypoint3D,localDataPath):
        with open(localDataPath,'a',newline='',encoding='utf-8-sig') as csv_file:
            Tags = ["Time","Hour"]
            info = {}
            info["Time"]=(self.framesCount-1)*0.033
            if(len(self.timeData)>0):
                info["Hour"] = self.timeData[self.framesCount-1][1]
            else:
                info["Hour"] = 0
            for i in range(len(keypoint3D)):
                Tag = [f"X{i}",f"Y{i}",f"Z{i}"]
                Tags += Tag
                info[Tag[0]]=keypoint3D[i][0]
                info[Tag[1]]=keypoint3D[i][1]
                info[Tag[2]]=keypoint3D[i][2]
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
            csv_writer.writerow(info)

    def WriteRows2(self,keypointRows,localDataPath):
        with open(localDataPath,'a',newline='',encoding='utf-8-sig') as csv_file:
            for i in range(len(keypointRows)):
                Tags = ["Time","Hour"]
                info = {}
                info["Time"]=(i)*0.033

                if(len(self.timeData)>0):
                    info["Hour"] = self.timeData[i][1]
                else:
                    info["Hour"] = 0
                points = int(len(keypointRows[0])/3)
                for j in range(points):
                    Tag = [f"X{j}",f"Y{j}",f"Z{j}"]
                    Tags += Tag
                    info[Tag[0]]=keypointRows[i][j*3]
                    info[Tag[1]]=keypointRows[i][(j*3)+1]
                    info[Tag[2]]=keypointRows[i][(j*3)+2]
                csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
                csv_writer.writerow(info)

    def imageFormat(self,originalImg):
        frame = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        newImg = QtGui.QPixmap.fromImage(image)
        return newImg
 
    def angleFrame(self,angle):
        with open(self.anglePath,'a',newline='',encoding='utf-8-sig') as csv_file:
            Tags = ["Time","Hour","Hip","Knee","Ankle"]
            info = {}
            info[Tags[0]] = (self.framesCount-1)* 0.033    
            if(len(self.timeData)>0):
                info["Hour"] = self.timeData[self.framesCount-1][1]
            else:
                info["Hour"] = 0
            info[Tags[2]] = angle[0]    
            info[Tags[3]] = angle[1]    
            info[Tags[4]] = angle[2]         
            csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
            csv_writer.writerow(info)

    def angleFrame2(self,angle):
        with open(self.anglePath,'a',newline='',encoding='utf-8-sig') as csv_file:
            for i in range(len(angle)):
                Tags = ["Time","Hour","Hip","Knee","Ankle"]
                info = {}
                info[Tags[0]] = (self.framesCount-1)* 0.033    
                if(len(self.timeData)>0):
                    info["Hour"] = self.timeData[self.framesCount-1][1]
                else:
                    info["Hour"] = 0
                info[Tags[2]] = angle[i][0]    
                info[Tags[3]] = angle[i][1]    
                info[Tags[4]] = angle[i][2]         
                csv_writer = csv.DictWriter(csv_file,fieldnames=Tags)
                csv_writer.writerow(info)

    def angleInitializer(self, mainPath,name):
        self.anglePath = mainPath + '/' + name + '-'+ self.fileAngle
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

    def setLabels(self):
        self.RbOne.setText(f'(0) {self.TbOne.text()}')
        self.RbTwo.setText(f'(1) {self.TbTwo.text()}')
        self.RbThree.setText(f'(2) {self.TbThree.text()}')
        self.RbFour.setText(f'(3) {self.TbFour.text()}')
        self.RbFive.setText(f'(4) {self.TbFive.text()}')
        self.RbSix.setText(f'(5) {self.TbSix.text()}')

        self.TbOne.setVisible(False)
        self.TbTwo.setVisible(False)
        self.TbThree.setVisible(False)
        self.TbFour.setVisible(False)
        self.TbFive.setVisible(False)
        self.TbSix.setVisible(False)
        self.PbNext.setVisible(False)

    def addLabels(self,labelPath):
        readedlabels = (ML.DataExtract(labelPath,0))[1:]
        readedlabels = np.asarray(readedlabels, dtype = np.int32, order ='C')

        print(self.video[0])
        print(len(readedlabels))

        self.videoLabels = np.concatenate((self.videoLabels,readedlabels[(self.wSize-1)*self.padding:]),axis=0)

    def addData(self,dataPath):
        readedData = (ML.DataExtract(dataPath,1))[1:]
        readedData = np.transpose(np.transpose(readedData)[17:])
        readedData = np.asarray(readedData, dtype = np.float64, order ='C')

        #####
        dataBase = []
        for row in readedData:
            dataBase.append(row)       #dataBase.append(row[17:])
        dataBase = np.array(dataBase)
        #####

        videoData = np.array(ML.formatLSTM2(wSize=self.wSize,data=dataBase, padding=self.padding))

        print(self.video[0])
        print(len(videoData))

        if(len(self.videoData) == 0):
            self.videoData = videoData
        else:
            self.videoData = np.concatenate((self.videoData,videoData),axis=0)

    def dataScaler(self, dataBase, mainPath, modelName):
        scaler = np.max(np.max(dataBase,axis=1),axis=0)
        scalerString = ''
        for i in scaler:
            scalerString += str(i) + ','
        scaledDataBase = dataBase/scaler

        scalerPath = mainPath + modelName + '-Scaler.csv'
        with open(scalerPath,'w',newline='',encoding='utf-8-sig') as csv_file:
            csv_file.write(scalerString)

        return scaledDataBase

    def trainModel(self):
        self.allOffUi()
        self.trainingUi()
        scaledDataBase = self.dataScaler(self.videoData, self.modelPath, self.modelName)
        scaledDataBase = scaledDataBase.reshape(-1,5,36)

        self.videoLabels, scaledDataBase = shuffle(self.videoLabels,scaledDataBase)

        trainSize = 0.8    # Defines the percentage of data use for training
        trainLen = int(len(scaledDataBase)*(trainSize))
        testLen = len(scaledDataBase)

        xTrain = scaledDataBase[0:trainLen]
        xTest = scaledDataBase[trainLen:testLen]

        yTrain = self.videoLabels[0:trainLen]
        yTest = self.videoLabels[trainLen:testLen]

        imagePath = self.modelPath + self.modelName
        Model = ML.buildLSTMClassificationModel(wSize=5,iSize=36,hSize=[64,32,8],classes=5,imagePath= imagePath)

        history = Model.fit(xTrain, yTrain, batch_size=64, epochs=100, validation_split=0.2, verbose=1)

        modelPath = self.modelPath + self.modelName + '.h5'
        Model.save(modelPath)

        ## Performance plot
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig((self.modelPath + self.modelName + '-History.png'),dpi=300)
        plt.show()

        plt.cla()

        ## Evaluation of the model
        testScores = Model.evaluate(xTest, yTest, verbose=2)
        print("Test loss:", testScores[0])
        print("Test accuracy:", testScores[1])

        ## Prediction of new samples
        predictions = Model.predict(xTest, batch_size=10, verbose=2)
        rounded_predictions = np.argmax(predictions, axis=-1)

        cm = confusion_matrix(y_true=yTest, y_pred=rounded_predictions)

        Labels = np.unique(self.videoLabels)
        cmPlotLabels = []
        for i in Labels:
            cmPlotLabels.append(f'class {i}')
        ML.plotConfusionMatrix(cm=cm,classes=cmPlotLabels, title='Confusion Matrix',save='true',path=(self.modelPath + self.modelName))

        self.allOffUi()
        self.initialVariables()
        self.initialUi()

    def closeEvent(self,event):
        result = QtWidgets.QMessageBox.question(self,
                      "Confirm Exit...",
                      "Are you sure you want to exit ?",
                      QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            self.flagStop = 1
            event.accept()
    
    def keyPressEvent(self, a0):
        # print(a0.key())
        if(a0.key() == 16777236):
            if(self.TbOne.isVisible()):
                self.setLabels()
            self.labelFrame()




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = Ui_Form()
    Form.show()
    sys.exit(app.exec_())
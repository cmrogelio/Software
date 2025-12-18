from ultralytics import YOLO
import cv2
import os

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

cv2.imshow("first",frame)


model_path = 'yolov8n-pose.pt' #Direccion del documento last.pt que se encuentra dentro de la carpeta de weights

# Load a model

enableGPU = True

modelSize = 5                                           # (1-5) While using live video the value must always be 1
YOLO_models = {1:'yolov8n-pose',2:'yolov8s-pose',3:'yolov8m-pose',4:'yolov8l-pose',5:'yolov8x-pose'} # YOLO models

if(enableGPU):
    model = YOLO(YOLO_models[modelSize] + ".pt")
else:
    modelPath = str(YOLO_models[modelSize])+"_int8_openvino_model/"
    if(not os.path.exists(modelPath)):
        model = YOLO(YOLO_models[modelSize] + ".pt")
        model.export(format="openvino",half=True, int8=True)           # Generates the optimized model
    model = YOLO(modelPath)

threshold = 0.7

while ret:


    results = model(frame, conf=threshold)[0]
    imageYOLO = results.plot()
    cv2.imshow('Yolo',imageYOLO)

    ret, frame = cap.read()

    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break

cap.release() 
cv2.destroyAllWindows()


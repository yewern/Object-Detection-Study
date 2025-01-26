from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#object
cap = cv2.VideoCapture(0)  # For Webcam
#cap = cv2.VideoCapture("../Videos/bikes.mp4")  # For Videos

#webcame size
cap.set(3, 1280) #width
cap.set(4, 720)  #height

#getting pre-trained model with weight, yolo version 8 nano
model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "person", "person", "person", "person", "person", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True,device='mps')
    #YW: adding device = 'mps' to run faster. Without 180ms, With 35ms
    for r in results:
        boxes = r.boxes #bounding box of each result
        for box in boxes:
            # Bounding Box: with image and box but without label
            '''
            x1, y1, x2, y2 = box.xyxy[0] #tensor value
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) #give each detected item a box
            cvzone.cornerRect(img, (x1, y1, w, h))
            '''

            # Bounding Box: with image and box but without label but different type of rect
            x1, y1, x2, y2 = box.xyxy[0]  # tensor value
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            #adding confidence level of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            #display label
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)


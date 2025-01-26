from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#object
#cap = cv2.VideoCapture(0)  # For Webcam
cap = cv2.VideoCapture("../Videos/people.mp4")  # For Videos

#webcame size
cap.set(3, 1280) #width
cap.set(4, 720)  #height

#getting pre-trained model with weight, yolo version 8 nano
model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

mask = cv2.imread("Mask.png")

#tracking: if the car comes in, it should remain its ID as it detected and not change
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#creating a line to count, as long as the ID passes through the line the count as 1
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True,device='mps')

    detections = np.empty((0, 5))

    #YW: adding device = 'mps' to run faster. Without 180ms, With 35ms
    for r in results:
        boxes = r.boxes #bounding box of each result
        for box in boxes:

            # Bounding Box: with image and box but without label but different type of rect
            x1, y1, x2, y2 = box.xyxy[0]  # tensor value
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            #show box when something detected regardless of what object
            #cvzone.cornerRect(img, (x1, y1, w, h))

            #adding confidence level of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            #display label
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            #Detector: detect desired class and confidence greater than desired amount
            if (currentClass == "person" and conf > 0.3):
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=1, thickness=1, offset=3) #text for the detector rectangle
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5) #detector the rectangle
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray)) #stack value together

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    #Tracker: Keep tracking the detected object, if the first object come into detection, it should remains as 1 in all the frames it exits
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255)) #blue line of rect

        #observed ID does not go in sequence, because there might be some other thing being detected and took up the value
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=1, offset=10 )

        # with the created line, if the center of the rectangle passes through the line then it should count as 1
        # center of the rectangle
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        #counting
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)


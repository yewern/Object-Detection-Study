from ultralytics import YOLO
import cv2

#device = torch.device('mps' if torch.backends.mps.is_available else'cpu')#

model = YOLO('yolov8n.pt')
results = model("images/img2.png", show=True)
cv2.waitKey(0)
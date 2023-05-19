from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("./runs/detect/train2/weights/best.pt")

results = model.predict( source="1", show=True, save_crop=True)



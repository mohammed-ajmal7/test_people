import ultralytics
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="D:\BE-CSE\Count-the-number-of-people-currently-in-the-room\data.yaml", epochs =15)
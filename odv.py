# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:36:09 2025

@author: Bagirathan
"""

import os
import argparse
from ultralytics import YOLO


import sys
sys.argv = [
    'yolov8_train.py',
    '--data', 'E:/Senthil/AI/6.Deep Learning/Object detector with Voice/data.yaml',
    '--model', 'yolov8n.pt', 
    '--epochs', '50'
]

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Path to data.yaml")
ap.add_argument("--model", default="yolov8n.pt", help="YOLOv8 base model")
ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
args = vars(ap.parse_args())

# Load the YOLOv8 model
print("[INFO] Loading YOLOv8 model...")
model = YOLO(args["model"])

  
print("[INFO] Starting training...")
results = model.train(
    data=args["data"],
    epochs=args["epochs"],
    imgsz=640,
    batch=16,
    name="object_detector_yolov8"
)

print("[INFO] Training complete.")
print("[INFO] Best model saved at: runs/detect/object_detector_yolov8/weights/best.pt")

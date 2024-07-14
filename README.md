



# Counting the number of people using OpenCV

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Software and Tools](#software-and-tools)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration File](#configuration-file)
6. [Model Setup](#model-setup)
7. [Training the Model](#training-the-model)
    - [Rename Trained Weights](#Rename-trained-weights)
    - [Model Inference](#model-inference)
7. [Explanation Code](#explanation-code)
    - [Python Code for Box Detection](#python-code-for-box-detection)
8. [Output Explanation](#output-explanation)
6. [Applications](#applications)
7. [Future Scope](#future-scope)
8. [Summary](#summary)


## Introduction

This project involves annotating a dataset containing images with people, training a  model for counting the number of people in a room, and using the trained model to detect number os people present in the room . The steps below detail the entire process from dataset preparation to model training and testing.

## Requirements

- **Python 3.x**
- **OpenCV 4.x**

## Software and Tools

- **Python:** Programming language used for the script.
- **OpenCV:** Library for computer vision tasks.
- 
## Dataset Preparation
[Labeling image](https://drive.google.com/drive/folders/1WHpExY04EewfeeqJdMxsMg6maO7FCz8_?usp=drive_link)
1. **Annotate Images**: Use `labelimg` to annotate the dataset containing images with boxes. Save the annotations in the appropriate format.


2. **Split Dataset**: Split the annotated dataset into training and validation sets, each containing `images` and `labels` folders.
   
```



3. Install OpenCV: Install the OpenCV package using the command:
```bash
pip install opencv-python
pip install opencv-contrib-python
```
4. Activate Virtual Environment: Activate your virtual environment using following commands.
1. Go to command prompt
2. Open the current directory that the project is stored
3. Run the following commands one by one :
   python -m venv myenv
   cd myenv
   cd Scripts
   Activate
  

6. Verify Installation: Check if PyTorch is installed and CUDA is available using the following commands:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

## Training the Model
Train the YOLOv8 model using the following command:

```bash
yolo task=detect mode=train epochs=30 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=3
```
This process might take some time. Upon completion, you will get a best.pt file located inside runs>detect>train>weights.


### Rename Trained Weights
Rename the best.pt file to yolov8m_custom.pt and move it to the root directory.

### Model Inference
To detect boxes in an image using the trained model, use the following command:

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg
```
## Explanation Code 
### Python Code for Box Detection
Use the following Python code to import the model yolov8m_custom.pt and detect boxes in an image, displaying the number of boxes detected:

```python
import cv2
import numpy as np

# Load the image
image = cv2.imread('box3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding or edge detection to highlight the boxes
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image and retrieve the hierarchy
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store valid contours (boxes)
valid_contours = []

# Define minimum and maximum threshold values for box size
min_area = 1000  # Adjust this value based on the minimum area of your boxes
max_area = 5000  # Adjust this value based on the maximum area of your boxes

# Iterate through all detected contours
for i, contour in enumerate(contours):
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # Calculate the aspect ratio
    aspect_ratio = float(w) / h
    
    # Filter contours based on area, aspect ratio, and hierarchy
    if area > min_area and area < max_area and aspect_ratio > 0.5 and aspect_ratio < 2.0:
        # Check if contour has no parent (top-level contour)
        if hierarchy[0][i][3] == -1:
            valid_contours.append(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Count the number of detected valid contours (boxes)
num_boxes = len(valid_contours)

# Display the result
cv2.putText(image, f'Number of Boxes: {num_boxes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow('Boxes Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of boxes detected
print(f'Number of boxes detected: {num_boxes}')
```

## Output Explanation
After running the inference code, the output will be an image with detected boxes highlighted by green rectangles. The number of detected boxes will be displayed both on the image and printed in the console. The contours detected by the algorithm are filtered based on their area, aspect ratio, and hierarchy to ensure only valid boxes are counted.

## Applications
The techniques and processes described in this project have several practical applications, including:

- **Automated Quality Control**: Detecting defects or missing components in manufacturing.
- **Logistics and Inventory Management**: Identifying and counting items in warehouses.
- **Surveillance and Security**: Monitoring and detecting objects in security footage.
- **Retail**: Managing stock and detecting product placement on shelves.

## Future Scope
Future enhancements and extensions to this project could include:

- **Multi-Class Detection**: Expanding the model to detect and classify multiple types of objects.
- **Real-Time Detection**: Implementing real-time detection using video feeds.
- **Improved Accuracy**: Fine-tuning the model and using more sophisticated data augmentation techniques to improve detection accuracy.
- **Deployment**: Creating a web or mobile application to deploy the model for practical use.


## Summary

This project demonstrates the complete pipeline of annotating a dataset, training a YOLOv8 model, and using the trained model to detect objects in images. By following the steps outlined, one can develop a custom object detection model tailored to specific needs, with various practical applications across different industries. The future scope suggests further improvements and extensions to enhance the model's capabilities and deployment options.


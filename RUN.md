# how to run this code

## clone this repo

    git clone --branch ajmal_1TJ21CS053 https://github.com/EduCollaborations/Count-the-number-of-people-currently-in-the-room.git

## Create Virtual Environment

    create a virtual environment
    python -m venv env
    cd env/Scripts
    source activate

## requirements

    pip install opencv-python
    pip install opencv-contrib-python
    pip install ultralytics
    pip install torch torchvision torchaudio

### run the 1st.py
This process might take some time. Upon completion, you will get a best.pt file located inside runs>detect>train>weights.

### Rename Trained Weights
Rename the best.pt file to yolov8m_custom.pt and move it to the root directory.

### Model Inference
To detect boxes in an image using the trained model, use the following command:

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg
```
change the source= (give the path of the valid directory)
the predicted files will be saved inside runs>detect>train>

### Python Code for Counting the number of people
    run the 2nd.py
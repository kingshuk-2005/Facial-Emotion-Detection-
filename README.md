# Real-Time Facial Emotion Detection using CNN
This project is a deep learning-based real-time facial emotion detection system built with TensorFlow, Keras, and OpenCV. It detects human facial expressions from a live webcam feed and classifies them into one of seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

Model Overview
The model is a Convolutional Neural Network (CNN) trained on grayscale 48x48 facial images using data augmentation techniques to improve generalization. It is designed to classify facial emotions from static images and real-time video frames.

Project Structure
facepro.py – Script to train the CNN model using preprocessed image data.

testmodel.py – Script to load the trained model and perform real-time emotion recognition using a webcam.

requirement.txt – Required dependencies for the project.

Requirements
Install the dependencies using:

bash
Copy
Edit
pip install -r requirement.txt
Usage
Train the model:

bash
Copy
Edit
python facepro.py
Run real-time emotion detection:

bash
Copy
Edit
python testmodel.py
Press q to quit the webcam window.

Dataset
The model expects the dataset to be structured as:

bash
Copy
Edit
data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   └── ... (other emotion folders)
├── test/
│   ├── Angry/
│   ├── Disgust/
│   └── ... (other emotion folders)
Features
Real-time webcam-based emotion detection

Lightweight CNN with good accuracy

Data augmentation for better training performance


import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Directory where processed images are stored
processed_images_dir = 'D:\The_final\processed_images'
details_dir = 'D:\The_final\processed_images'
os.makedirs(details_dir, exist_ok=True)

# Ensure the details directory exists
if not os.path.exists(details_dir):
    os.makedirs(details_dir)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to train the recognizer
def train_recognizer():
    faces = []
    labels = []
    label_map = {}

    # Loop through processed images and extract faces
    for label, image_name in enumerate(os.listdir(processed_images_dir)):
        image_path = os.path.join(processed_images_dir, image_name)
        
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Detect faces in the image
        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        # Loop through detected faces
        for (x, y, w, h) in faces_detected:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label)
            label_map[label] = image_name
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    recognizer.save('trained_model.yml')  # Save the trained model for later use

    return label_map

# Load the trained model
def load_trained_model():
    recognizer.read('trained_model.yml')  # Load the trained model
    return recognizer

# Function to detect face from the webcam and match with stored images
def detect_and_match_face():
    cap = cv2.VideoCapture(0)
    
    # Get the current date to use as the log filename
    current_date = datetime.now().strftime('%Y-%m-%d')
    details_file_path = os.path.join(details_dir, f"{current_date}.csv")

    # Initialize the DataFrame to store details
    if os.path.exists(details_file_path):
        df = pd.read_csv(details_file_path)
    else:
        df = pd.DataFrame(columns=["Date", "Image Name"])

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Crop the detected face region
            face = gray[y:y+h, x:x+w]
            
            # Predict the label (ID) of the detected face
            label, confidence = recognizer.predict(face)
            
            name = f"Unknown (Confidence: {confidence:.2f})"
            
            if confidence < 100:  # You can adjust this threshold as needed
                raw_label = label_map.get(label, "Unknown")
                name = os.path.splitext(raw_label)[0]
            # Draw rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Only save the details to CSV if the name is not "Unknown"
            if name != "Unknown":  # Skip "Unknown" faces
                date = datetime.now().strftime('%Y-%m-%d')
                
                # Check if this name and date are already logged
                if not ((df["Date"] == date) & (df["Image Name"] == name)).any():
                    new_row = pd.DataFrame({"Date": [date], "Image Name": [name]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(details_file_path, index=False)  # Save to CSV file
                
        # Show the frame with detected faces and labels
        cv2.imshow('Webcam Face Recognition', frame)
        
        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Train the recognizer using the processed images
label_map = train_recognizer()

# Once trained, you can now run the face detection and recognition on webcam
detect_and_match_face()
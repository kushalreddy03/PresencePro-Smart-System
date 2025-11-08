import cv2
import os

# Directory to save the captured images
image_dir = 'D:\The_final\images'  # Adjust the path as needed

# Ensure the directory exists, create it if it doesn't
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Counter for image file naming
image_counter = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame in a window
    cv2.imshow("Webcam", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Save the frame if the 's' key is pressed
    if key == ord('s'):
        # Ask for a custom name for the image
        image_name = input("Enter a name for the image (without extension): ")
        
        # Ensure the image name is not empty
        if image_name.strip():
            # Construct the image file path with the entered name
            image_path = os.path.join(image_dir, f"{image_name}.jpg")
            
            # Save the image
            cv2.imwrite(image_path, frame)
            print(f"Saved image to {image_path}")
        else:
            print("Invalid image name. Please enter a valid name.")

    # Exit if the 'q' key is pressed
    if key == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
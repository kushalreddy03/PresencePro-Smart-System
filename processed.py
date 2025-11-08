import cv2
import os

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process a single image
def process_image(image_path, save_dir):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert the image to grayscale (Haar cascades work on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Log the face locations (for debugging)
    if len(faces) > 0:
        print(f"Detected faces in {image_path}: {faces}")
    else:
        print(f"No faces detected in {image_path}.")

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the processed image to the specified directory with the custom name
    output_path = os.path.join(save_dir, os.path.basename(image_path))  # Keeps the original filename
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

# Directory where the images are located
image_dir = 'D:\The_final\images'  # Adjust to your directory where images are stored

# Directory to save the processed images
save_dir = 'D:\The_final\processed_images'  # Directory for processed images

# Ensure the save directory exists, create if it doesn't
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List all images in the directory with custom names
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

# Process each image in the directory
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    process_image(image_path, save_dir)

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Image processing completed.")
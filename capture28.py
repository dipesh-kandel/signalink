import cv2
import os

# Create a folder named "images" if it doesn't exist
folder = './cnn/imageHello28Test'
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize webcam
camera = cv2.VideoCapture(0)

# Function to capture image inside the box, resize, and save it with a unique filename
def capture_image(roi, folder, counter):
    # Resize the ROI to 28x28
    resized_roi = cv2.resize(roi, (28, 28))
    
    filename = os.path.join(folder, f'captured_image_{counter}.jpg')
    cv2.imwrite(filename, resized_roi)
    print("Image saved as", filename)
    return counter + 1

# Main loop
counter = 1  # Counter for generating unique filenames
while True:
    # Read frame from webcam
    ret, frame = camera.read()
    
    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Define box size and position
    box_size = 200
    box_x = frame.shape[1] - box_size - 50  # Right side position
    box_y = 50
    
    # Extract the region of interest (ROI) inside the box
    roi = frame[box_y:box_y + box_size, box_x:box_x + box_size]
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Invert the colors inside the box (background as white, object as black)
    inverted_thresholded_roi = cv2.bitwise_not(gray_roi)
    
    # Replace the ROI in the original frame with the inverted thresholded ROI
    frame[box_y:box_y + box_size, box_x:box_x + box_size] = cv2.cvtColor(inverted_thresholded_roi, cv2.COLOR_GRAY2BGR)
    
    # Draw a box on the frame
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)
    
    # Display the mirrored frame
    cv2.imshow('Webcam', frame)
    
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    
    # If 'a' key is pressed, capture, resize, and save the image inside the box with a unique filename
    if key == ord('a'):
        counter = capture_image(roi, folder, counter)
    
    # If 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

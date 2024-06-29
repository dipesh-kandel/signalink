import numpy as np
import cv2
import time
import pyttsx3
import threading
from tensorflow.keras.models import load_model
from customtkinter import *
from runprogram import open_main_ui
from imagepath import *
from colors import *
from PIL import Image,ImageTk
is_playing =True
#*****************************Exits the current Window*****************# 
def exit ():
    root.destroy()
    open_main_ui()

#*************************Closes the Camera*****************# 
def on_closing():
    global is_playing
    is_playing = False
    canvas.image = None
    canvas.update()

#*****************change between Fullscreen and window **********#
def toggle_fullscreen(event=None):
    state = not root.attributes('-fullscreen')
    root.attributes('-fullscreen', state)

def main():
    global is_playing
    is_playing = True
    if is_playing :
        close_btn.place(relx = 0.96, rely=0.065,anchor = "center")
    model = load_model("./model/cnn.h5")

    # Initialize webcam
    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    # Function to preprocess the image
    def preprocess_image(img):
        # Convert image to grayscale
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert colors
        inverted = cv2.bitwise_not(gray)
        # Resize to 28x28
        resized = cv2.resize(inverted, (28, 28))
        # Reshape to match model input shape
        preprocessed = resized.reshape(1, 28, 28, 1)
        # Normalize pixel values
        preprocessed = preprocessed / 255.0
        return preprocessed

    engine = pyttsx3.init()
    def speak_text(text):
        engine.say(text)
        engine.runAndWait()

    def getLetter(result):
        classLabels = {0: 'Hello', 1: 'ILoveYou', 2: 'Yes',3:'blank'}
        try:
            return classLabels[result]
        except KeyError:
            return "Unknown"

# Function to classify the image
    def classify_image(img):
        preprocessed_img = preprocess_image(img)
        prediction = model.predict(preprocessed_img)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        return class_index, confidence

# Main loop
    newclass_label = ""
    class_index = 3
    while True:
        if not is_playing: 
            close_btn.place(relx = 50, rely=50,anchor = "center") # ESC
            break
        # Read frame from webcam
        ret, frame = cap1.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Mirror the frame
        frame = cv2.flip(frame, 1)

        box_size = 200
        box_x = frame.shape[1] - box_size - 50  # Right side position
        box_y = 50
        # Extract the region of interest (ROI) inside the box
        roi = frame[box_y:box_y + box_size, box_x:box_x + box_size]
        
        # Classify the image inside the box
        # Get class label
        class_label = getLetter(class_index)
        
        
        if class_label !=newclass_label:
            threading.Thread(target=speak_text, args=(class_label,)).start()
            newclass_label=class_label
        class_index, confidence = classify_image(roi)
        # Display class label and confidence on the frame
        cv2.putText(frame, f"Class: {class_label}", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (box_x, box_y + box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw a box on the frame
        debug_img =cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)
        
        # Display the frame
        img = Image.fromarray(debug_img)
        imgtk = ImageTk.PhotoImage(image=img)
        frame.flags.writeable = False
        
        canvas.create_image(0,0,anchor=NW,image=imgtk)
        canvas.image = imgtk
        canvas.update()
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' key is pressed, exit the loop
        if key == ord('q'):
            break

    # Release resources
    # cap1.release()
    # cv2.destroyAllWindows()

#************** ui design starts *********************#

#********** main window configuration ************#
root = CTk()
root.title("Speech to Sign")
root.configure(fg_color = main_bg_color)
root.geometry("1920x1080")
root.after(0,root.wm_state,"zoomed")
root.attributes('-fullscreen', True)
root.bind('<Escape>', toggle_fullscreen)

#*****************************Back To Home button *******************************************#
exit_image = CTkImage(light_image=Image.open(exit_path),
                                  dark_image=Image.open(exit_path),
                                  size=(30, 30))
exit_btn = CTkButton(master=root,hover_text_color=button_bg_color,text_color=button_bg_color,
                     width=40,height=40,text="",fg_color="transparent",
                     image=exit_image,hover=False,command=exit)
exit_btn.place(relx=0.075,rely=0.05,anchor="ne")
label5 = CTkLabel(root,text="Back To Home Page",text_color=button_bg_color)
label5.place(relx=0.095,rely=0.09,anchor="ne")

#***************frame Starts**********************************************#
frame = CTkFrame(master=root,width=980,height=960,fg_color = main_bg_color)
frame.place(relx = 0.5,rely = 0.5,anchor="center")

canvas = CTkCanvas(frame,width=960,height=540,bg=frame_bg_color,borderwidth=0,highlightthickness=0)
canvas.place(relx=0.1,rely=0.3)
logo_image = CTkImage(light_image=Image.open(logo_path),
                                  dark_image=Image.open(logo_path),
                                  size=(200, 75))
image_label = CTkLabel(frame,image=logo_image,text="")
image_label.place(relx = 0.5,rely = 0.15,anchor = "center")
label3 =CTkLabel(master= frame,text="SIGN TO SPEECH TRANSLATION USING CNN",text_color=brown_color,font=("Arial",-24))
label3.place(relx = 0.5 , rely=0.2,anchor = "center")
btn1 =CTkButton(master=frame,text="Open Camera",hover_text_color=white_color,corner_radius=20,
                fg_color=button_bg_color,border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=white_color,width=150,height=40,
                font=("Arial",-14),command=main)
btn1.place(relx = 0.5 , rely=0.25,anchor = "center")
close_image = CTkImage(light_image=Image.open(close_path),
                                  dark_image=Image.open(close_path),
                                  size=(30, 30))

close_btn =CTkButton(master=canvas,hover_text_color=button_bg_color,width=40,height=40,text="",
                     fg_color="transparent",image=close_image,hover=False,command=on_closing)

#***************** Frame Ends *************************************************************#
root.mainloop()
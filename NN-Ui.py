import numpy as np
import cv2
import pickle
import time
import pyttsx3
import threading
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
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model['W1'], model['b1'], model['W2'], model['b2']

    def classify_image(image, W1, b1, W2, b2):
        # Preprocess the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to convert the ROI to black and white
        _, thresholded_roi = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        
        # Invert colors to have black object on white background
        inverted_thresholded_roi = cv2.bitwise_not(gray_image)

        # Resize image to match model input size
        resized_image = cv2.resize(inverted_thresholded_roi, (28, 28))
        

        # Normalize pixel values
        normalized_image = resized_image / 255.0
        
        # Flatten image
        X = normalized_image.flatten().reshape(1, -1).T
        
        # Make prediction
        prediction = make_predictions(X, W1, b1, W2, b2)
        return prediction

    # Function to make predictions using the model
    def make_predictions(X, W1, b1, W2, b2):
        Z1 = np.dot(W1, X) + b1
        A1 = np.maximum(Z1, 0)  # ReLU activation
        Z2 = np.dot(W2, A1) + b2
        A2 = np.exp(Z2 - np.max(Z2)) / np.sum(np.exp(Z2 - np.max(Z2)), axis=0)  # Softmax activation
        predictions = np.argmax(A2, axis=0)
        return predictions

    # Load the trained model
    W1_saved, b1_saved, W2_saved, b2_saved = load_model('./model/ann1.pkl')

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Open the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    start_time = time.time()
    label = 'Label: '
    prev_prediction = None

    # Function to speak the text asynchronously
    def speak_text(text):
        engine.say(text)
        engine.runAndWait()

    while True:
        if not is_playing: 
            close_btn.place(relx = 50, rely=50,anchor = "center") # ESC
            break

        ret, frame = cap.read()

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Create a rectangular box of 200x200 on the right side of the screen
        box_size = 200
        box_x = frame.shape[1] - box_size - 50  # Right side position
        box_y = 50

        # Extract the region of interest (ROI) inside the box
        roi = frame[box_y:box_y + box_size, box_x:box_x + box_size]

        # Draw a box on the frame
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)

        current_time = time.time()
        if current_time - start_time >= 2:  # Update label every 5 seconds
            try:
                # Classify the image inside the box
                prediction = classify_image(roi, W1_saved, b1_saved, W2_saved, b2_saved)

                # Update label only if prediction changes
                if prediction != prev_prediction:
                    if prediction == 0:
                        text = "Hello"
                    elif prediction == 1:
                        text = "ILoveYou"
                    else:
                        text = "yes"
                    label = f'Label: {text}'
                    prev_prediction = prediction
                    
                    # Speak the text asynchronously
                    threading.Thread(target=speak_text, args=(text,)).start()

                start_time = time.time()  # Reset the timer

            except Exception as e:
                print(f"Error: {e}")

        # Display the label above the box
        debug_img = cv2.putText(frame, label, (box_x, box_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with the classification label
        img = Image.fromarray(debug_img)
        imgtk = ImageTk.PhotoImage(image=img)
        frame.flags.writeable = False
    
        canvas.create_image(0,0,anchor=NW,image=imgtk)
        canvas.image = imgtk
        canvas.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
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
label3 =CTkLabel(master= frame,text="SIGN To SPEECH TRANSLATION USING FNN",text_color=brown_color,font=("Arial",-24))
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
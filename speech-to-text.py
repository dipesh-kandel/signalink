import speech_recognition as sl
import os
import cv2
from customtkinter import *
from runprogram import open_main_ui
from PIL import Image,ImageTk
from colors import *
from imagepath import *

#*****************************Exits the current Window*****************# 
def exit ():
    root1.destroy()
    open_main_ui()

#*****************change between Fullscreen and window **********#
def toggle_fullscreen(event=None):
    state = not root1.attributes('-fullscreen')
    root1.attributes('-fullscreen', state)

#*********** plays video corresponding to word *************#
def play_video(word):
    # Check if the video file exists for the given word
    video_file_path = f"video/{word}.mp4"
    
    if not os.path.exists(video_file_path):
        cap = cv2.VideoCapture("video/idle.mp4")
    else:
        cap = cv2.VideoCapture(video_file_path)
    # Play the video
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(308,548))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(0,0,image=photo,anchor = NW)
        canvas.image = photo
        canvas.update()


    cap.release()

#************ Recognizes speech and coverts to text ******************# 
def speech():  
        
        label2.configure(text="")
        label2.update()
        sr = sl.Recognizer()
        with sl.Microphone() as source2:
            btn1.configure(text="Silent Please")
            btn1.update()
            sr.adjust_for_ambient_noise(source2,duration=2)

            btn1.configure(text="Speak Now Please")
            btn1.update()
            audio2 = sr.listen(source2)
            try:
                text = sr.recognize_google(audio2)
                text = text.lower()
                label1.configure(text= f"You said: {text.upper()} ",fg_color=button_bg_color,padx=4,pady=4)
                label1.update()
                keywords = list()
                keywords = text.split()
                for keyword in keywords:
                    play_video(keyword)
                canvas.image = None 
                canvas.update()
                label1.configure(text="",fg_color="transparent")
                label1.update()  
            except sl.UnknownValueError:
                print("Speech Recognition could not understand audio")
                label2.configure(text="Sorry, Couldn't Understand You")
                label2.update()
                # time.sleep(2)
            except sl.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                label2.configure(text="Speech Recognition error")
                label2.update()
            finally:
                btn1.configure(text="Speech to Sign")
                btn1.update()


#****************** ui design *******************************#

#********** main window configuration ************#         
root1 = CTk()
root1.title("SPEECH TO SIGN")
root1.configure(fg_color =main_bg_color)
root1.geometry("1920x1080")
root1.after(0,root1.wm_state,"zoomed")
root1.attributes('-fullscreen', True)
root1.bind('<Escape>', toggle_fullscreen)
#************Back To Home button *****************#
exit_image = CTkImage(light_image=Image.open(exit_path),
                                  dark_image=Image.open(exit_path),
                                  size=(30, 30))
exit_btn = CTkButton(master=root1,hover_text_color=button_bg_color,text_color=button_bg_color,width=40,height=40,text="",
                     fg_color="transparent",image=exit_image,hover=False,command=exit)
exit_btn.place(relx=0.075,rely=0.05,anchor="ne")
label5 = CTkLabel(root1,text="Back To Home Page",text_color=button_bg_color)
label5.place(relx=0.095,rely=0.09,anchor="ne")
#***************** Frame Starts *************************#
frame = CTkFrame(master=root1,width=600,height=960,fg_color = frame_bg_color)
frame.place(relx = 0.5,rely = 0.5,anchor="center")

canvas = CTkCanvas(frame,width=310,height=550,bg=frame_bg_color,borderwidth=0,highlightthickness=0)
canvas.place(relx=0.3,rely=0.3)
logo_image = CTkImage(light_image=Image.open(logo_path),
                                  dark_image=Image.open(logo_path),
                                  size=(200, 75))
image_label = CTkLabel(frame,image=logo_image,text="")
image_label.place(relx = 0.5,rely = 0.15,anchor = "center")
label3 =CTkLabel(master= frame,text="SPEECH TO SIGN TRANSLATION",text_color=brown_color,font=("Arial",-24))
label3.place(relx = 0.5 , rely=0.2,anchor = "center")
label1 = CTkLabel(master=frame,text="",text_color=white_color,font=("Arial",-18))
label1.place(relx = 0.5 , rely=0.8,anchor = "center")
label2 = CTkLabel(master=frame,text="",text_color=button_bg_color,font=("Arial",-14))
label2.place(relx = 0.5 , rely=0.285,anchor = "center")

mic_image = CTkImage(light_image=Image.open(mic_path),
                                  dark_image=Image.open(mic_path),
                                  size=(30, 30))
btn1 =CTkButton(master=frame,text="Open Mic",hover_text_color=white_color,corner_radius=20,
                fg_color=button_bg_color,border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=white_color,width=150,height=40,
                font=("Arial",-14),image=mic_image,command=speech)
btn1.place(relx = 0.5 , rely=0.25,anchor = "center")
#***************** Frame Ends *************************#




root1.mainloop()

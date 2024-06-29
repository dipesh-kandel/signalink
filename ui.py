from customtkinter import *
from PIL import Image
from runprogram import open_sign_to_speech,open_speech_to_sign,open_analysis,open_nn_ui
from imagepath import *
from colors import *


#***************** exits app ****************#
def exitmain():
    app.destroy()

#*************** Function to call speech to sign window *********#
def speech_to_sign():
    exitmain()
    open_speech_to_sign()

#*************** Function to call sign to speech window *********#
def sign_to_speech():
    exitmain()
    open_sign_to_speech() 


def analysis():
    exitmain()
    open_analysis()

def nn_ui():
    exitmain()
    open_nn_ui()

#*****************change between Fullscreen and window **********#
def toggle_fullscreen(event=None):
    state = not app.attributes('-fullscreen')
    app.attributes('-fullscreen', state)


#********** main ui design ******************#


#********** main window configuration ************#
app = CTk()
app.geometry("1920x1080")
app.title("SignLink")
app.configure(fg_color =main_bg_color)
app.after(0,app.wm_state,'zoomed')
app.attributes('-fullscreen',True)
app.bind('<Escape>', toggle_fullscreen)
#************ Exit button *****************#
exit_image = CTkImage(light_image=Image.open(exit_path),size=(30, 30))
exit_btn = CTkButton(master=app,hover_text_color=button_bg_color,text_color=button_bg_color,
                     width=40,height=40,text="",
                     fg_color="transparent",image=exit_image,hover=False,command=exitmain)
exit_btn.place(relx=0.975,rely=0.05,anchor="ne")
label5 = CTkLabel(app,text="Exit",text_color=button_bg_color)
label5.place(relx=0.965,rely=0.09,anchor="ne")

#**********************frame starts***************************#
frame0 = CTkFrame(app)    
frame0.configure(fg_color = frame_bg_color,width = 500,height = 900) 
frame0.place(relx = 0.5,rely = 0.5,anchor="center")
logo_image = CTkImage(light_image=Image.open(logo_path),size=(320, 120))
image_label = CTkLabel(frame0,image=logo_image,text="")
image_label.place(relx = 0.5,rely = 0.2,anchor = "center")
btn0 =CTkButton(master=frame0,text="Sign to Speech Using FNN",hover_text_color=white_color,
                corner_radius=20,fg_color='transparent',border_color=button_bg_color,
                border_width=2 ,hover_color=button_bg_color,text_color=black_color,
                width=150,height=40,font=("Arial",-14),command=nn_ui)
btn0.place(relx = 0.525,rely = 0.4,anchor = "center")
btn1 =CTkButton(master=frame0,text="Sign to Speech Using CNN",hover_text_color=white_color,corner_radius=20,
                fg_color='transparent',border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=black_color,width=150,height=40,
                font=("Arial",-14),command=sign_to_speech)
btn1.place(relx = 0.525,rely = 0.45,anchor = "center" )

btn2 =CTkButton(master=frame0,text="Speech to Sign",hover_text_color=white_color,
                corner_radius=20,fg_color='transparent',border_color=button_bg_color,
                border_width=2 ,hover_color=button_bg_color,text_color=black_color,
                width=150,height=40,font=("Arial",-14),command=speech_to_sign)
btn2.place(relx = 0.525,rely = 0.5,anchor = "center")

btn3 =CTkButton(master=frame0,text="Analysis of Two Model",hover_text_color=white_color,
                corner_radius=20,fg_color='transparent',border_color=button_bg_color,
                border_width=2 ,hover_color=button_bg_color,text_color=black_color,
                width=150,height=40,font=("Arial",-14),command=analysis)
btn3.place(relx = 0.525,rely = 0.55,anchor = "center")
love_image = CTkImage(light_image=Image.open(love_path),size=(344, 195.6))
image_label = CTkButton(frame0,image=love_image,text="",fg_color="transparent",hover_text_color="white",
                        hover=False,border_color=button_bg_color,border_width=2)
image_label.place(relx = 0.525,rely = 0.775,anchor = "center")
hand1_image = CTkImage(light_image=Image.open(hand1_path),size=(160, 106.6))
image_label = CTkButton(app,image=hand1_image,text="",fg_color="transparent",hover_text_color="white",hover=False)
image_label.place(relx = 0.95,rely = 0.935,anchor = "center")
hand2_image = CTkImage(light_image=Image.open(hand2_path),size=(160, 106.6))
image_label = CTkButton(app,image=hand2_image,text="",fg_color="transparent",hover_text_color="white",hover=False)
image_label.place(relx = 0.02,rely = 0.935,anchor = "center")
#**************************frame end *********************************#

app.mainloop()
import os
import sys
from subprocess import call

current_directory = os.path.dirname(os.path.abspath(__file__))

def open_script(script_name):
    script_path = os.path.join(current_directory, script_name)
    try:
        call([sys.executable, script_path])
    except Exception as e:
        print(f"Error executing {script_name}: {e}")

def open_sign_to_speech():
    open_script("CNN_UI.py")

def open_speech_to_sign():
    open_script("speech-to-text.py")

def open_main_ui():
    open_script("ui.py")

def open_analysis():
    open_script("analysisui.py")

def open_nn_ui():
    open_script("NN-UI.py")




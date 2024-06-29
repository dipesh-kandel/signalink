from customtkinter import *
from colors import *
# from training import train_model, evaluate_model,accuracy,line_plot,print_confusion_matrix,predict_model,metrices1

from training_cnn import train_model,load_data,print_confusion_matrix,predict_model,line_plot,preprocess_data,evaluate_model,metrices1,acc_plot
from indexWith3Label import train_model2,acc,make_predictions,get_accuracy,print_confusion_matrix1,metrices,line_plot1,acc_plot1
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from runprogram import open_main_ui
from PIL import Image,ImageTk
from imagepath import *

#*****************************Exits the current Window*****************# 
def exit ():
    root.destroy()
    open_main_ui()

#*****************change between Fullscreen and window **********#
def toggle_fullscreen(event=None):
    state = not root.attributes('-fullscreen')
    root.attributes('-fullscreen', state)


# ********************Box Plot*********************************#
def create_boxplot(model1_accuracy, model2_accuracy):
    accuracy_scores = [model1_accuracy, model2_accuracy]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')

    box_props = dict(facecolor='lightgreen', color='blue')
    whisker_props = dict(color='black')
    median_props = dict(color='red')
    flier_props = dict(marker='o', markerfacecolor='green', markersize=8, linestyle='none')

    ax.boxplot(accuracy_scores, labels=['CNN', 'FNN'], patch_artist=True, boxprops=box_props, whiskerprops=whisker_props, medianprops=median_props, flierprops=flier_props)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Accuracy Between Two Models')
    ax.grid(True)
    return fig

# ********************Violin Plot*********************************#
def create_violinplot(model1_accuracy, model2_accuracy):
    # Combine the accuracy scores into a single list
    accuracy_scores = [model1_accuracy, model2_accuracy]
    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')
    # Create the violin plot
    ax.violinplot(accuracy_scores, showmeans=True, showextrema=True)

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Accuracy Between Two Models')

    # Add x-axis ticks
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['CNN', 'FNN'])

    # Show the plot
    ax.grid(True)

    # Return the figure and axis
    return fig

# ********************Bar Plot*********************************#
def Barplots(accuracy, precision, recall, f1score):
    models = ["CNN", "FNN"]

    bar_width = 0.2
    r1 = range(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Create a new figure
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')
    # Create bar plot for accuracy
    ax.bar(r1, accuracy, color='skyblue', width=bar_width, edgecolor='grey', label='Accuracy')

    # Create bar plot for precision
    ax.bar(r2, precision, color='salmon', width=bar_width, edgecolor='grey', label='Precision')

    # Create bar plot for recall
    ax.bar(r3, recall, color='lightgreen', width=bar_width, edgecolor='grey', label='Recall')

    ax.bar(r4, f1score, color='gold', width=bar_width, edgecolor='grey', label='f1-score')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(models))])
    ax.set_xticklabels(models)

    # Add labels and title
    ax.set_ylabel('Performance Metrics')
    ax.set_title('Comparison of Model Performance')

    # Add legend
    ax.legend()

    # Show the plot
    ax.grid(True)

    # Return the figure
    return fig

# *******************************GUI Class ***********************************************************#
class TrainingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Training Model")
        self.exit_image = CTkImage(light_image=Image.open(exit_path),
                                  dark_image=Image.open(exit_path),
                                  size=(30, 30))
        self.exit_btn = CTkButton(master=root,hover_text_color=button_bg_color,text_color=button_bg_color,
                     width=40,height=40,text="",fg_color="transparent",
                     image=self.exit_image,hover=False,command=exit)
        self.exit_btn.place(relx=0.075,rely=0.05,anchor="ne")
        self.label5 = CTkLabel(master,text="Back To Home Page",text_color=button_bg_color)
        self.label5.place(relx=0.095,rely=0.09,anchor="ne")
        self.logo_image = CTkImage(light_image=Image.open(logo_path),
                                  dark_image=Image.open(logo_path),
                                  size=(200, 75))
        image_label = CTkLabel(master,image=self.logo_image,text="")
        image_label.place(relx = 0.25,rely = 0.2,anchor = "center")
        label3 =CTkLabel(master= master,text="Analysis of Model",text_color=brown_color,font=("Arial",-24))
        label3.place(relx = 0.25, rely=0.3,anchor = "center")
        self.btn3 =CTkButton(master,text="Load Data",hover_text_color=white_color,corner_radius=20,
                fg_color='transparent',border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=black_color,width=100,height=30,
                font=("Arial",-14),command=self.load)
        self.btn3.place(relx = 0.275,rely = 0.35,anchor = "ne" )
        # ************************** CNN ******************************************
        self.label_CNN = CTkLabel(master,text="For CNN ",text_color=button_bg_color)
        self.label_CNN.place(relx=0.15,rely=0.4,anchor="ne")
        self.label_epoch = CTkLabel(master,text="Epochs: ",text_color=button_bg_color)
        self.label_epoch.place(relx=0.095,rely=0.45,anchor="ne")
        self.entry_epoch = CTkEntry(master,
                                    height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_epoch.place(relx=0.200,rely=0.45,anchor="ne")

        self.label_batch = CTkLabel(master,text="Batch: ",text_color=button_bg_color)
        self.label_batch.place(relx=0.095,rely=0.5,anchor="ne")
        self.entry_batch = CTkEntry(master,
                                    height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_batch.place(relx=0.200,rely=0.5,anchor="ne")
        self.label_learning = CTkLabel(master,text="Learning Rate: ",text_color=button_bg_color)
        self.label_learning.place(relx=0.095,rely=0.55,anchor="ne")
        self.entry_lr = CTkEntry(master,
                                    height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_lr.place(relx=0.200,rely=0.55,anchor="ne")
        self.test_acc = CTkLabel(master,text="Test Accuracy of CNN: ",text_color=button_bg_color)
        self.test_acc.place(relx=0.195,rely=0.6,anchor="ne")
        self.train_acc = CTkLabel(master,text="Train Accuracy of CNN: ",text_color=button_bg_color)
        self.train_acc.place(relx=0.195,rely=0.65,anchor="ne")
        # **************************FNN******************************************
        self.label_nn = CTkLabel(master,text="For FNN ",text_color=button_bg_color)
        self.label_nn.place(relx=0.35,rely=0.4,anchor="ne")
        self.label_epoch1 = CTkLabel(master, text="Epoch:",text_color=button_bg_color)
        self.label_epoch1.place(relx=0.295,rely=0.45,anchor="ne")

        self.entry_epoch1 = CTkEntry(master,
                                     height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_epoch1.place(relx=0.400,rely=0.45,anchor="ne")
        self.label_batch1 = CTkLabel(master, text="Batch:",text_color=button_bg_color)
        self.label_batch1.place(relx=0.295,rely=0.5,anchor="ne")

        self.entry_batch1 = CTkEntry(master,
                                     height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_batch1.place(relx=0.400,rely=0.5,anchor="ne")
        self.label_learning1 = CTkLabel(master,text="Learning Rate: ",text_color=button_bg_color)
        self.label_learning1.place(relx=0.295,rely=0.55,anchor="ne")
        self.entry_lr1 = CTkEntry(master,
                                    height=40,width=150,
                                    corner_radius=50,
                                    text_color=button_bg_color,
                                    fg_color=(button_bg_color,frame_bg_color))
        self.entry_lr1.place(relx=0.400,rely=0.55,anchor="ne")
        self.test_acc1 = CTkLabel(master,text="Test Accuracy of FNN:",text_color=button_bg_color)
        self.test_acc1.place(relx=0.4,rely=0.6,anchor="ne")
        self.train_acc1 = CTkLabel(master,text="Train Accuracy of FNN:",text_color=button_bg_color)
        self.train_acc1.place(relx=0.4,rely=0.65,anchor="ne")
        self.btn1 =CTkButton(master,text="train",hover_text_color=white_color,corner_radius=20,
                fg_color='transparent',border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=black_color,width=100,height=30,
                font=("Arial",-14),command=self.train)
        self.btn1.place(relx = 0.275,rely = 0.72,anchor = "ne" )
        self.analysis_type =CTkComboBox(master,width=175,height=30,corner_radius=50,fg_color=frame_bg_color,border_color=button_bg_color,
                                        font=("Arial",-12),button_color=button_bg_color,dropdown_hover_color=button_bg_color,border_width=2,
                                        dropdown_fg_color=(button_bg_color,frame_bg_color),button_hover_color=button_bg_color,dropdown_text_color=black_color,text_color=black_color,
                                        values=["Confusion Matrix","Loss Plot","Accuracy Plot","Bar Plot","Box Plot","Violin Plot",],justify="center",command=self.select_analysis)
        self.analysis_type.set("Select an Option") 
        self.analysis_type.place(relx = 0.3,rely = 0.77,anchor = "ne")
        self.label1 = CTkLabel(master, text="",text_color=button_bg_color)
        self.label1.place(relx = 0.30,rely = 0.82,anchor = "ne")

    # ********** embed fig in canvas tkinter ui **********#
    def embed_diagram(self,fig):
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.7, rely=0.3,anchor="center")
        
    # ********** embed fig in canvas tkinter ui **********#
    def embed_diagram1(self,fig):
        self.canvas1 = FigureCanvasTkAgg(fig, master=root)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().place(relx=0.7, rely=0.7,anchor="center")

    # ********** embed fig in canvas tkinter ui **********#
    def embed_diagram2(self):
        self.canvas1.figure.clf()
        self.canvas1.draw()

    # ****************** dropdown menu ****************#
    def select_analysis(self,event):
        selected_analysis = self.analysis_type.get()
        print(selected_analysis)
        if selected_analysis == "Confusion Matrix":
            y_pred = predict_model(model,self.x_test)
            fig = print_confusion_matrix(y_pred,self.y_test)
            self.embed_diagram(fig)
            model2_predictions_test = make_predictions(W1,b1,W2,b2)
            fig2 =print_confusion_matrix1(model2_predictions_test)
            self.embed_diagram1(fig2)

        elif selected_analysis == "Loss Plot":
            fig4 = line_plot(history,epochs)
            fig6 = line_plot1(epochs1)
            self.embed_diagram(fig4)
            self.embed_diagram1(fig6)
        elif selected_analysis == "Bar Plot":
            fig3 = Barplots(accuracy_list,precision_list,recall_list,f1score_list)
            self.embed_diagram(fig3)
            self.embed_diagram2()
        elif selected_analysis == "Box Plot":
            fig4 = create_boxplot(model1_train_acc_list,modelnn_train_acc_list)
            self.embed_diagram(fig4)
            self.embed_diagram2()
        elif selected_analysis == "Violin Plot":
            fig5 = create_violinplot(model1_train_acc_list,modelnn_train_acc_list)
            self.embed_diagram(fig5)
            self.embed_diagram2()
        elif selected_analysis == "Accuracy Plot":
            fig5=acc_plot(history,epochs)
            fig6=acc_plot1(epochs1)
            self.embed_diagram(fig5)
            self.embed_diagram1(fig6)
   

    def complete_function(self,text):
        self.label1.configure(text=text)
        self.label1.update()
    # *********** load data function **************** #
    def load(self):
        self.label1.configure(text="Data is Loading.....")
        self.label1.update()
        self.train_data, self.test_data = load_data()
        self.x_train, self.x_test,self.y_train, self.y_test, self.unique_labels = preprocess_data(self.train_data, self.test_data)
        self.complete_function("Data Loaded Successfully now Train the Data")
    # *********** training function *************#
    def train(self):
        self.label1.configure(text="Model is training.....")
        self.label1.update()
        global epochs,history,model,modelnn_train_acc_list,model1_train_acc_list,accuracy_list,precision_list,recall_list,f1score_list,epochs1
        epochs = int(self.entry_epoch.get()) 
        batch_size = int(self.entry_batch.get())
        learning_rate = float(self.entry_lr.get())
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1score_list =[]

        #* CNN
        history,model = train_model(self.x_train,self.y_train,self.x_test,self.y_test,epochs, batch_size,learning_rate)
        model1_train_acc_list =history.history['accuracy']
        train_acc = model1_train_acc_list[-1]
        model1_acc= evaluate_model(model,self.test_data)
        y_pred = predict_model(model,self.x_test)
        
        acc1,precision1,recall1,f1score1 = metrices1(y_pred,self.y_test)
        accuracy_list.append(model1_acc)
        precision_list.append(precision1)
        recall_list.append(recall1)
        f1score_list.append(f1score1)
        acc_str = "{:.2%}".format(model1_acc)
        train_acc_str ="{:.2%}".format(train_acc)
        print(accuracy_list)
        self.test_acc.configure(text="Test Accuracy of CNN: "+acc_str)
        self.test_acc.update()
        self.train_acc.configure(text="Train Accuracy of CNN: "+train_acc_str)
        self.train_acc.update()

        #* model2 FNN
        epochs1 = int(self.entry_epoch1.get()) 
        batch1 = int(self.entry_batch1.get())
        learning_rate1 = float(self.entry_lr1.get())
        self.label1.configure(text="Model is Training")
        self.label1.update()
        global W1,b1,W2,b2
        W1,b1,W2,b2 =train_model2(learning_rate1,epochs1,batch1)
        model2_acc,modelnn_train_acc_list = acc()
        print(model2_acc)
        model2_predictions_test = make_predictions(W1,b1,W2,b2)
        model2_test_acc = get_accuracy(model2_predictions_test)
        acc2,precision2,recall2,f1score2 = metrices(model2_predictions_test)
        accuracy_list.append(acc2)
        precision_list.append(precision2)
        recall_list.append(recall2)
        f1score_list.append(f1score2)
        acc_str1 = "{:.2%}".format(model2_test_acc)
        self.test_acc1.configure(text="Test Accuracy of FNN: "+acc_str1)
        self.test_acc1.update()
        train_acc_str1 = "{:.2%}".format(model2_acc)
        self.train_acc1.configure(text="Train Accuracy of FNN: "+train_acc_str1)
        self.train_acc1.update()

        self.complete_function("Model has been trained")

    
if __name__ == "__main__":
    root = CTk()
    root.title("Analysis")
    root.configure(fg_color = main_bg_color)
    root.geometry("1920x1080")
    root.after(0,root.wm_state,"zoomed")
    root.attributes('-fullscreen', True)
    root.bind('<Escape>', toggle_fullscreen)
    gui = TrainingGUI(root)
    root.mainloop()
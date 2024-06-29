import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

def load_data():
    train = pd.read_csv('./data/image28.csv')
    test = pd.read_csv('./data/image28Test.csv')
    return train, test

# train ,test = load_data()
def preprocess_data(train, test):
    labels = train['Label'].values
    unique_labels = np.unique(labels)
    
    train.drop('Label', axis=1, inplace=True)
    images = train.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    images = np.array([i.flatten() for i in images])
    
    label_binrizer = LabelBinarizer()
    labels = label_binrizer.fit_transform(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_train, x_test, y_train, y_test, unique_labels

def build_model(lr):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(3, activation='softmax'))
    
    sgd = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

# x_train,x_test, y_train, y_test, unique_labels = preprocess_data()
def train_model(x_train, y_train, x_test, y_test, epochs=10, batch_size=32,lr=0.001):
    model = build_model(lr)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history,model

def evaluate_model(model, test_data):
    test_labels = test_data['Label']
    test_data.drop('Label', axis=1, inplace=True)
    test_images = test_data.values
    test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
    test_images = np.array([i.flatten() for i in test_images])
    label_binrizer = LabelBinarizer()
    test_labels = label_binrizer.fit_transform(test_labels)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    y_pred = model.predict(test_images)
    accuracy = accuracy_score(test_labels, y_pred.round())
    return accuracy

def predict_model(model,X_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    label_binrizer = LabelBinarizer()
    y_pred = label_binrizer.fit_transform(y_pred)
    return y_pred


def print_confusion_matrix(y_pred, y_true):
    y_true_flat = [label.argmax() for label in y_true]  # Flatten y_true to a 1D list
    y_pred_flat = np.argmax(y_pred, axis=1)  # Flatten y_pred to a 1D array

    labels = sorted(list(set(y_true_flat)))  # Get unique labels
    cmx_data = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    fig.set_facecolor('#ddd8bf')
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true_flat)), 0)
    ax.set_title("Confusion Matrix of CNN")

    print('Classification Report')
    print(classification_report(y_true_flat, y_pred_flat))
    return fig


def acc_plot(model, epoch):
    train_acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    
    epochs = range(1, epoch+1)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    fig.set_facecolor('#ddd8bf')
    colors = ['#1f77b4', '#ff7f0e'] 

    ax.plot(epochs, train_acc, label='Training Accuracy', color=colors[0])

    ax.plot(epochs, val_acc, label='Validation Accuracy', color=colors[1])
   
    # Add labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy of CNN Model')

    ax.legend()
    return fig



def line_plot(model, epoch):
    train_loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(1, epoch+1)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')
    colors = ['#1f77b4', '#ff7f0e']

    ax.plot(epochs, train_loss, label='Training Loss', color=colors[0], marker='o')
    
    ax.plot(epochs, val_loss, label='Validation Loss', color=colors[1], marker='o')
   
    # Add labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss of CNN Model')

    ax.legend()
    return fig

def metrices1(y_pred,y_test):
    acc = metrics.accuracy_score(y_test,y_test)
    precision = metrics.precision_score(y_test,y_test,average='weighted')
    recall = metrics.recall_score(y_test,y_pred,average='weighted')
    f1 = metrics.f1_score(y_test,y_pred,average='weighted')

    return acc,precision,recall,f1

def inference_single_image(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255
    image = np.reshape(image, (1, 28, 28, 1))
    prediction = model.predict(image)
    return prediction

def main():
    train_data, test_data = load_data()
    x_train, x_test, y_train, y_test, unique_labels = preprocess_data(train_data, test_data)
    history,model = train_model(x_train, y_train, x_test, y_test,epochs=5,batch_size=32,lr =0.001)
    y_pred = predict_model(model,x_test)
    confusion_matrix_fig = print_confusion_matrix(y_pred,y_test)
    line_fig = line_plot(history,epoch=10)
    model.save("./model/cnn.h5")
    print("Model Saved")
    a,p,r,f = metrices1(y_pred,y_test)
    fig = acc_plot(history,epoch=10)
    # plt.show()
    accuracy = evaluate_model(model,test_data)
    print("Accuracy:", accuracy)
    image_path = './cnn/imageHello28Test/captured_image_100.jpg'
    prediction = inference_single_image(model, image_path)
    print("Prediction:", prediction)
    
if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
accuracy_train = 0
train_acc_list = []
train_loss_list =[]
validation_loss_list =[]
validation_acc_list =[]
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    np.random.shuffle(data)
    return data

def preprocess_data(data):
    m, n = data.shape
    X = data[:, 1:].T / 255.0  # Normalize input data
    Y = data[:, 0]
    return X, Y

data = load_data('./data/image28.csv')
X_train, Y_train = preprocess_data(data)
X_train, X_val, Y_train, Y_val = train_test_split(X_train.T, Y_train, test_size=0.3, random_state=42)
X_train=X_train.T
X_val = X_val.T
data2 = load_data('./data/image28Test.csv')
X_test,Y_test = preprocess_data(data2)

def init_params(input_size, output_size):
    W1 = np.random.rand(10, input_size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(output_size, 10) - 0.5
    b2 = np.random.rand(output_size, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # Subtracting max(Z) for numerical stability
    return exp_Z / np.sum(exp_Z, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    num_classes = len(np.unique(Y))
    m = X.shape[1]
    one_hot_Y = one_hot(Y, num_classes)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)  # ReLU derivative
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2    
    return W1, b1, W2, b2

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y=Y_val):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X_train, Y_train, X_val, Y_val, alpha, iterations, batch_size):
    train_acc_list.clear()
    validation_loss_list.clear()
    train_loss_list.clear()
    validation_acc_list.clear()
    W1, b1, W2, b2 = init_params(X_train.shape[0], len(np.unique(Y_train)))
    num_samples_train = X_train.shape[1]
    num_samples_val = X_val.shape[1]
    for i in range(iterations):
        for batch_start in range(0, num_samples_train, batch_size):
            batch_end = min(batch_start + batch_size, num_samples_train)
            X_batch = X_train[:, batch_start:batch_end]
            Y_batch = Y_train[batch_start:batch_end]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Calculate training loss and accuracy
        global accuracy_train
        _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train)
        loss_train = compute_loss(A2_train, Y_train)
        predictions_train = get_predictions(A2_train)
        accuracy_train = get_accuracy(predictions_train, Y_train)
        train_acc_list.append(accuracy_train)

        # Calculate validation loss and accuracy
        _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_val)
        loss_val = compute_loss(A2_val, Y_val)
        predictions_val = get_predictions(A2_val)
        accuracy_val = get_accuracy(predictions_val, Y_val)
        validation_loss_list.append(loss_val)
        train_loss_list.append(loss_train)
        validation_acc_list.append(accuracy_val)
        print(f"Iteration {i}: Training Loss = {loss_train:.4f}, Training Accuracy = {accuracy_train:.2f}, Validation Loss = {loss_val:.4f}, Validation Accuracy = {accuracy_val:.2f}")
    
    return W1, b1, W2, b2


def acc():
    return accuracy_train,train_acc_list


def acc_plot1(epoch):
    epochs = range(1, epoch+1)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    fig.set_facecolor('#ddd8bf')
    colors = ['#1f77b4', '#ff7f0e'] 
    ax.plot(epochs, train_acc_list, label='Training Accuracy', color=colors[0])
    ax.plot(epochs, validation_acc_list, label='Validation Accuracy', color=colors[1])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Accuracy of FNN Model')
    ax.legend()
    return fig

def line_plot1(epoch):
    
    epochs = range(1, epoch+1)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')

    colors = ['#1f77b4', '#ff7f0e']  

  
    ax.plot(epochs, train_loss_list, label='Training Loss', color=colors[0], marker='o')

    ax.plot(epochs, validation_loss_list, label='Validation Loss', color=colors[1], marker='o')
   

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss of FNN Model')

    ax.legend()
    return fig

def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(W1, b1, W2, b2,X[:, index, None])
    label = Y[index]
    print("Prediction:", prediction)
    print("Label:", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    

def make_predictions(W1, b1, W2, b2,X=X_val):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions




def train_model2(lr=0.01,epoch=10,batch_size=32):
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train,X_val,Y_val, alpha=lr, iterations=epoch,batch_size=batch_size)
    return W1,b1,W2,b2




def save_model(W1, b1, W2, b2, file_path):
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def compute_loss(A2, Y):
    num_samples = Y.shape[0]
    one_hot_Y = one_hot(Y, A2.shape[0])
    loss = -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / num_samples
    return loss


def print_confusion_matrix1(y_pred,y_true=Y_val):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots()
    fig.set_size_inches(8,4.5)
    fig.set_facecolor('#ddd8bf')
    ax.set_title("Confusion Matrix of FNN")
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)

    print('Classification Report')
    print(classification_report(Y_val, y_pred))
    return fig

def metrices(prediction,Y=Y_val):
    acc = metrics.accuracy_score(Y,prediction)
    precision = metrics.precision_score(Y,prediction,average='weighted')
    recall = metrics.recall_score(Y,prediction,average='weighted')
    f1 = metrics.f1_score(Y,prediction,average='weighted')

    return acc, precision,recall,f1

def main():
    
    W1,b1,W2,b2= train_model2()
    acc1 = acc()
    
    save_model(W1, b1, W2, b2, './model/ann1.pkl')
    # line_plot(10)
    predictions_test = make_predictions(W1=W1, b1=b1, W2=W2, b2=b2,X=X_val)
    print_confusion_matrix1(predictions_test)
    acc1,precision,recall,f1 = metrices(predictions_test)
    predictions_test = make_predictions(W1=W1, b1=b1, W2=W2, b2=b2,X=X_test)
    accuracy_test = get_accuracy(predictions_test, Y=Y_test)
    acc_plot1(10)
    line_plot1(10)
    plt.show()
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model['W1'], model['b1'], model['W2'], model['b2']

    W1_saved, b1_saved, W2_saved, b2_saved = load_model('./model/ann1.pkl')
    test_prediction(7, X_test, Y_test, W1_saved, b1_saved, W2_saved, b2_saved)


if __name__ == "__main__":
    main()


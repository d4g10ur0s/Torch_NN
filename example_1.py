import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# utilities
import pandas as pd
from sklearn.model_selection import train_test_split
# plotting
import matplotlib.pyplot as plt
# custom modules
from Mushroom import Mushroom
from perceptron import Perceptron

# plot the data
def get_color(label, prediction):
    if label == 0:
        # class 0
        return 'black'
    elif label == 1:
        # class 1
        return 'yellow'
    else:
        raise ValueError(f"Invalid label value: {label}")
    # Misclassification logic (assuming threshold of 0.5 for binary classification)
    if (label == 0 and prediction > 0.5) or (label == 1 and prediction < 0.5):
        # misclassification
        return 'red'
    else:
        # correct classification color
        return get_color(label, prediction)
# training funciton
def train_model(model, optimizer, criterion, train_loader, device, epoch):
    # use the specified device
    model.to(device)
    # set the model to training mode
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        labels = torch.tensor(labels).float()
        loss = criterion(outputs, labels.unsqueeze(1))
        # backprobagate the error
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += labels.size(0)
    # print results and return average loss
    average_loss = total_loss / total_samples
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")
    return average_loss
# test function
def test_model(model, criterion, test_loader, device):
    # use the specified device
    model.to(device)
    # set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    # disable gradient calculation during evaluation
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = torch.tensor(labels).float()
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()
            total_samples += labels.size(0)
        # calculate accuracy for binary classification
        # threshold for binary classification
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
    average_loss = total_loss / total_samples
    accuracy = correct / total_samples
    print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return average_loss, accuracy, predictions

'''

    ***  Main Script  ***

'''
# csv file path
csv_file = "/home/d4gl0s/torch_nn/data/PoisonousMushrooms/data.csv"
dataset = Mushroom(csv_file)
# split data into training set and test set
test_size = 0.1
train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)
# define data loaders and batch size
batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# hyperparameters
learning_rate = float(input('Set learning rate : '))
momentum = float(input('Set momentum : '))
epochs = int(input('Set number of epochs : '))
# get the number of input features from the dataset (assuming all features are used)
input_dim = len(dataset[0][0])
# create the model and optimizer
model = Perceptron(input_dim, 1)  # 1 output for binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.BCELoss()
# training loop
avgLoss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # train the model
    average_loss = train_model(model, optimizer, criterion, train_loader, device, t)
    # visualize training loss
    avgLoss.append(average_loss)
# plot the average loss
plt.plot(range(epochs), avgLoss, label='Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
# test after training
test_loss, accuracy, test_predictions = test_model(model, criterion, test_loader, device)
# extract weights from linear layer
weights = model.linear.weight.squeeze()
# extract bias
bias = model.linear.bias.item()
# calculate separation line equation (y = mx + b)
decision_boundary = torch.linspace(-1, 1, 100)
'''
*** there was an error  ***
'''
decision_boundary = decision_boundary.to(device)
separation_line = weights[0] * decision_boundary + bias
# plot data points with colors based on the get_color function
plt.scatter(test_features[:, 0].numpy(), test_features[:, 1].numpy(), c=map(get_color, test_labels.numpy(), test_predictions.numpy()))
# plot separation line
plt.plot(decision_boundary.numpy(), separation_line.numpy(), color='green', label='Separation Line')
# add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Mushroom Classification Results')
plt.legend()
plt.grid(True)
plt.show()

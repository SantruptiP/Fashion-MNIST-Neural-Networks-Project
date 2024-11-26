import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt
'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

batch_size = 64


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net()
conv_net = Conv_Net()



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

# Variables to store loss for plotting
ffn_losses = []
cnn_losses = []


'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = 10

for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn
        inputs = inputs.view(-1, 28 * 28)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()
    average_loss_ffn = running_loss_ffn / len(trainloader)
    ffn_losses.append(average_loss_ffn)

    print(f"Epoch {epoch+1}, Feedforward Network Loss: {running_loss_ffn / len(trainloader)}")
    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = 10

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
    average_loss_cnn = running_loss_cnn / len(trainloader)
    cnn_losses.append(average_loss_cnn)

    print(f"Epoch {epoch+1}, Convolutional Network Loss: {running_loss_cnn / len(trainloader)}")
    print(f"Training loss: {running_loss_cnn}")

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:
        inputs, labels = data
        # Evaluation for Feedforward Network
        inputs_ffn = inputs.view(-1, 28 * 28)
        outputs_ffn = feedforward_net(inputs_ffn)
        _, predicted_ffn = torch.max(outputs_ffn, 1)
        total_ffn += labels.size(0)
        correct_ffn += (predicted_ffn == labels).sum().item()

        # Evaluation for Convolutional Network
        outputs_cnn = conv_net(inputs)
        _, predicted_cnn = torch.max(outputs_cnn, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

        '''
        
            YOUR CODE HERE

        '''

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:


Check the instructions PDF. You need to generate some plots. 

'''
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_example_images(model, loader, network_type):
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            if network_type == "FFN":
                inputs = inputs.view(-1, 28 * 28)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_img, correct_pred, correct_label = inputs[i], predicted[i].item(), labels[i].item()
                else:
                    incorrect_img, incorrect_pred, incorrect_label = inputs[i], predicted[i].item(), labels[i].item()
                if 'correct_img' in locals() and 'incorrect_img' in locals():
                    return correct_img, correct_pred, correct_label, incorrect_img, incorrect_pred, incorrect_label

correct_ffn, pred_ffn_correct, label_ffn_correct, incorrect_ffn, pred_ffn_incorrect, label_ffn_incorrect = get_example_images(feedforward_net, testloader, "FFN")
correct_cnn, pred_cnn_correct, label_cnn_correct, incorrect_cnn, pred_cnn_incorrect, label_cnn_incorrect = get_example_images(conv_net, testloader, "CNN")


epochs = range(1, num_epochs_ffn + 1)


plt.figure()
plt.plot(epochs, ffn_losses, label="Feedforward Network Loss")
plt.plot(epochs, cnn_losses, label="Convolutional Network Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.show()

# Visualization for Prediction Examples
def show_example_images(correct_img, correct_pred, correct_label, incorrect_img, incorrect_pred, incorrect_label, title):
    correct_img = correct_img.view(28, 28).cpu()
    incorrect_img = incorrect_img.view(28, 28).cpu()
    
    plt.figure(figsize=(8, 4))
    
    # Correct prediction
    plt.subplot(1, 2, 1)
    plt.imshow(correct_img, cmap="gray")
    plt.title(f"Correct: {class_labels[correct_label]}\nPredicted: {class_labels[correct_pred]}")
    
    # Incorrect prediction
    plt.subplot(1, 2, 2)
    plt.imshow(incorrect_img, cmap="gray")
    plt.title(f"True: {class_labels[incorrect_label]}\nPredicted: {class_labels[incorrect_pred]}")
    
    plt.suptitle(title)
    plt.show()

show_example_images(correct_ffn, pred_ffn_correct, label_ffn_correct, incorrect_ffn, pred_ffn_incorrect, label_ffn_incorrect, "Feedforward Network Predictions")
show_example_images(correct_cnn, pred_cnn_correct, label_cnn_correct, incorrect_cnn, pred_cnn_incorrect, label_cnn_incorrect, "Convolutional Network Predictions")

'''

YOUR CODE HERE

'''

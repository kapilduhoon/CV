## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #output size dimensions: (32, 220, 220) 
        self.conv1 = nn.Conv2d(1, 32, 5)
       
        # output size dimension (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        
        # ouput size dimension (64, 108, 108)
        # after max pooling (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # output dimension (128, 52, 52)
        # after max pooling (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # output dimension (256, 24, 24)
        # after max pooling (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 3)
       
        # output dimension (512, 12, 12)
        # max pool (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 136)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
       
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout5(x)
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import imageio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 43)
        """

        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        
        # Spatial transformer localization-network
        """
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        """
        self.localization = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ELU(),
            nn.Conv2d(50, 100, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ELU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        #print(111)
        xs = self.localization(x)
        #print(222)
        
        xs = xs.view(-1, 100 * 4 * 4)
        #print(333)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        """
        #print(x.shape)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.conv3_bn(x)
        #print("size", x.shape)
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

dataset_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/rebalanced_data/"
output_dir =  "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/transformed_rebalanced_data/"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)

transform = transforms.Resize((32,32), interpolation=Image.NEAREST)

model.load_state_dict(torch.load("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/SPTN_weights.pt"))
model.eval()

for categorie in os.listdir(dataset_dir):
    if not os.path.exists(output_dir + categorie):
            os.makedirs(output_dir + categorie)
    for img_path in os.listdir(dataset_dir + categorie):
        #img = Image.open(dataset_dir + categorie + "/" + img_path)#cv2.imread(dataset_dir + categorie + "/" + img_path)
        img = cv2.imread(dataset_dir + categorie + "/" + img_path)
        #print(img.shape)

        #img = cv2.resize(img, (32,32), cv2.INTER_NEAREST)
        #print(img.shape)
        img_tensor = transforms.ToTensor()(img) #from_numpy(img).to(device)
        img_tensor = transform(img_tensor)
        #print(img_tensor.shape)
        out = model.stn(img_tensor.unsqueeze(0).to(device))
        #out = out.numpy().transpose((1, 2, 0))
        img_transformed = out[0].cpu().detach().numpy().transpose((1, 2, 0))
        img_transformed = img_transformed * 255
        
        #print(type(img_transformed))
        #print(
        cv2.imwrite(output_dir + categorie + "/" + img_path, img_transformed)
        print(categorie)
        
        
    

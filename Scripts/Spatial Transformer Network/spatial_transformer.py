from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile, Image
import cv2

print(1)
plt.ion()   # interactive mode
print(2)
print(3)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 43)
        

        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 44)
        
        # Spatial transformer localization-network

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



def train(epoch):
    model.train()
    #print(00)
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(11)
        data, target = data.to(device), target.to(device)
        #print(22)
        optimizer.zero_grad()
        #print(33)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


print(4)
train_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/rebalanced_data"
test_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/test" 
#batch_size = 16
epochs = 1
#opener = urllib.request.build_opener()
#opener.addheaders = [('User-agent', 'Mozilla/5.0')]
#urllib.request.install_opener(opener)
print(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)

#model.load_state_dict(torch.load("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/SPTN_weights.pt"))
print("loaded")
optimizer = optim.SGD(model.parameters(), lr=0.001)
print(6)

# Training dataset
"""
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)

"""
train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.Resize((32,32), Image.NEAREST), #transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            #normalize,
        ])),
        batch_size= 16, shuffle=True)
print(7)
# Test dataset
"""
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)
"""
test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            #transforms.Scale(256),
            transforms.Resize((32,32), Image.NEAREST), #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #normalize,
        ])),
        batch_size=16, shuffle=True)
print(8)
#for (x,y) in test_loader:
#    print(x)
#    print(y)


for epoch in range(1, epochs + 1):
    print(9)
    train(epoch)
    test()

#torch.save(model.state_dict(), "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/SPTN_weights.pt")

#Visualize the STN transformation on some input batch
#visualize_stn()

#plt.ioff()
#plt.show()


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import os

train_data_dir = "../data/train"
val_data_dir = "../data/val"
test_data_dir = "../data/test"

classes = os.listdir(train_data_dir)

train_transforms = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])
val_transforms = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])
test_transforms = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(train_data_dir, transform= train_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32)
val_data = datasets.ImageFolder(val_data_dir, transform= val_transforms)
valloader = torch.utils.data.DataLoader(val_data, batch_size = 32)
test_data = datasets.ImageFolder(test_data_dir, transform= test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bcm1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bcm2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.bcm3 = nn.BatchNorm2d(num_features=32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.bcm4 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(in_features=64*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=len(classes))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                #                 nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bcm1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bcm2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.bcm3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.bcm4(x)

        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc3(x)
        return x

def train(epochs = 100):
    model = Model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print("epoch: ", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            #forward + backward + optimizer
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print training statistics
            running_loss += loss.item()
            if i % 32 == 31:  #print every 64 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, running_loss/32))
                running_loss = 0
    print("finished training")

    current_sub_dir = os.listdir("checkpoint")
    new_save_sub_dir = os.path.join("checkpoint", str(len(current_sub_dir) + 1))
    os.makedirs(new_save_sub_dir, exist_ok=False)
    save_path = os.path.join(new_save_sub_dir, "model.pth")
    torch.save(model.state_dict(), save_path)

def test():
    weight_path = "checkpoint/1/model.pth"
    model = Model()
    model.load_state_dict(torch.load(weight_path))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on test images: %d %%" % (100*correct/total))

if __name__ == "__main__":
    train(epochs=50)
    #test()




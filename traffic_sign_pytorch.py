
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time
import pandas
from PIL import Image
import numpy as np


class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class TrafficSignDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, root_dir, transforms=transforms.Compose([transforms.ToTensor()])):
        self.data_frame = pandas.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        path = self.data_frame['Path'][index]
        img_path = self.root_dir / path
        label = torch.Tensor([self.data_frame['ClassId'][index]]).long()

        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)
        
        sample = (image, label)
        return sample



class ModelTrainer:
    def __init__(self, net, train_dataloader, test_dataloader, learning_rate=0.001, num_epochs=5, num_classes=2):
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), learning_rate)
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = num_classes

    def train(self):
        self.net.to(self.device)
        self.net.train()
        
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            start_time = time.time()


            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                labels = torch.squeeze(labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                # outputs = torch.argmax(outputs)
                
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                if i % 200 == 0:
                    print(f"Loss: {loss}")
                

            end_time = time.time()
            print(f"Epoch took {end_time - start_time} seconds.")

        print('Finished Training')

    def test(self):
        self.net.eval()

        class_correct = list(0. for i in range(self.classes))
        class_total = list(0. for i in range(self.classes))

        overall_correct = 0
        overall_total = 0
        loss = 0.0

        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                labels = torch.squeeze(labels)
                outputs = self.net(images)
                # outputs = torch.argmax(outputs)
                loss += self.loss_criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                overall_total += labels.size(0)
                overall_correct += (predicted == labels).sum().item()
                correct = (predicted == labels).squeeze()
                
                for i, label in enumerate(labels):
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        overall_accuracy = overall_correct / overall_total
        print(f'Overall Accuracy: {overall_accuracy}')

        for i in range(self.classes):
            print('Accuracy of %5s : %2d %%' % (
                i, 100 * class_correct[i] / class_total[i]))




if __name__ == '__main__':
    ### Hyperparameters ###
    batch_size = 32
    num_epochs = 3
    ######################


    data_root_path = Path("C:/Arbeit/datasets/traffic_sign")
    train_path = data_root_path / "Train"
    test_path = data_root_path / "Test"
    
    """transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])"""

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])


    # train_set = torchvision.datasets.ImageFolder(str(train_path), transform=transforms)
    train_set = TrafficSignDataset(data_root_path / "Train.csv", data_root_path, transforms=transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    test_set = TrafficSignDataset(data_root_path / "Test.csv", data_root_path, transforms=transforms)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # model = SmallConvNet()
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 43)

    trainer = ModelTrainer(model, train_dataloader, test_dataloader, num_epochs=num_epochs, num_classes=43)
    trainer.train()
    print("Training finished. Starting evaluation.")
    trainer.test()
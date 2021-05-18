
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time


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
        x = self.fc3(x)
        return x

class ModelTrainer:
    def __init__(self, net, train_dataloader, test_dataloader, learning_rate=0.0001, num_epochs=5):
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), learning_rate)
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.net.to(self.device)
        self.net.train()
        
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            start_time = time.time()

            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                if i % 200 == 0:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            end_time = time.time()
            print(f"Epoch took {end_time - start_time} seconds.")

        print('Finished Training')

    def test(self):
        self.net.eval()

        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))

        overall_correct = 0
        overall_total = 0
        loss = 0.0

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += self.loss_criterioncriterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                overall_total += labels.size(0)
                overall_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        overall_accuracy = overall_correct / overall_total
        print(f'Overall Accuracy: {overall_accuracy}')

        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))




if __name__ == '__main__':
    ### Hyperparameters ###
    batch_size = 32
    ######################


    data_root_path = Path("C:/Arbeit/datasets/traffic_sign")
    train_path = data_root_path / "Train"
    test_path = data_root_path / "Test"
    transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(str(train_path), transform=transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = SmallConvNet()
    trainer = ModelTrainer(model, train_dataloader, None)
    trainer.train()
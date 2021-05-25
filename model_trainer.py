from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, net, train_dataloader, test_dataloader, loss_criterion=nn.BCELoss(), learning_rate=0.0001, num_epochs=5, num_classes=2):
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_criterion = loss_criterion
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

                """if self.classes > 2:
                    outputs = torch.argmax(outputs, dim=1)"""
                
                
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                if i % 200 == 0:
                    print(f"Loss: {loss}")
                

            end_time = time.time()
            print(f"Epoch {epoch + 1} took {end_time - start_time} seconds.")

        print('Finished Training')

    def multiclass_test(self):
        """
        Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        """
        self.net.eval()

        class_correct = list(0. for i in range(self.classes))
        class_total = list(0. for i in range(self.classes))

        overall_correct = 0
        overall_total = 0
        loss = 0.0

        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # labels = torch.squeeze(labels)
                outputs = self.net(images)
                if self.classes > 2:
                    outputs = torch.argmax(outputs)

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

    def binary_test(self):
        self.net.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                outputs = torch.round(outputs)
                
                all_preds.extend(outputs.cpu())
                all_labels.extend(labels.cpu())

        
        accuracy = accuracy_score(all_preds, all_labels)
        print(f"Overall Accuracy: {accuracy}")
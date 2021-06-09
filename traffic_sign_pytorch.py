
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

from model_trainer import ModelTrainer
from datasets import TrafficSignDataset
from models import FullyConnectedNet, SmallConvNet

### Hyperparameters ###
batch_size = 32
num_epochs = 5
lr = 0.0001
######################


data_root_path = Path("C:/Arbeit/datasets/traffic_sign")
train_path = data_root_path / "Train"
test_path = data_root_path / "Test"

"""transforms = transforms.Compose([
    transforms.Grayscale(),
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

# model = FullyConnectedNet()

trainer = ModelTrainer(model, train_dataloader, test_dataloader, loss_criterion=nn.CrossEntropyLoss(), learning_rate=lr, num_epochs=num_epochs, num_classes=43)
trainer.train()
print("Training finished. Starting evaluation.")
trainer.multiclass_test()
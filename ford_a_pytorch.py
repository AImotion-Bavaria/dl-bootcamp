import numpy as np
from datasets import readucr
import torch
import time

from model_trainer import ModelTrainer
from models import TimeSeriesConvModel, FullyConnectedNet

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

### Hyperparameters ###
batch_size = 128# benchmarks with 128
num_epochs = 10
######################

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


print("Starting training of random forest classifier")
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy of Random Forest: {accuracy}")



x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# model = TimeSeriesConvModel()
model = FullyConnectedNet(input_dim=500, num_classes=1)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


trainer = ModelTrainer(model, train_dataloader, test_dataloader, num_epochs=num_epochs)
start = time.time()
trainer.train()
end = time.time()
print(f"Training finished, Took {end - start} seconds.")
trainer.binary_test()
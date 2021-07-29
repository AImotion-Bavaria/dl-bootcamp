import torch
import pandas
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

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

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
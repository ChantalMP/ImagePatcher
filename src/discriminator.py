from torch import nn
from torch.functional import F
import torch
import random

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        #input: 256x256x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        #256x256x16 -> pool: 128,128,16
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        #128x128x16 -> pool: 64,64,16
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        #64x64x16 -> pool: 32,32,16
        self.fc1 = nn.Linear(32*32*16, 128)
        #128
        self.fc2 = nn.Linear(128, 1)
        #1

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = x.view(-1, 32*32*16);
        x = self.fc1(x)
        x = F.relu(x)

        y = F.sigmoid(self.fc2(x))
        return y


    def prepare_data(self, originals, generated):

        x = originals+generated
        y = [1 for _ in range(len(originals))] +  [0 for _ in range(len(generated))]

        c = list(zip(x, y))
        random.shuffle(c)
        x,y = zip(*c)

        x = torch.FloatTensor(x)
        x = x.transpose(2,3).transpose(1,2)
        y = torch.FloatTensor(y)

        return x,y

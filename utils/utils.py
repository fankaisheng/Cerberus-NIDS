import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import sys
import numpy as np
from typing import Dict
from scipy import stats
import random
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import h5py
from tqdm import tqdm
import warnings
import pickle
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PCAP_DATASET(Dataset):
    def __init__(self, file):
        self.items = torch.zeros(0,1024)
        self.label = torch.zeros(0)
        label_list = ['Benign', 'Bot', 'DDoS', 'DoS', 'Patator', 'PortScan', 'WebAttack']
        with open(file, 'r', encoding='utf-8') as json_file:
            buf = json.load(json_file)
            for i,key in enumerate(label_list):
                if key in buf.keys():
                    self.items = torch.cat([self.items,torch.tensor(buf[key])],dim=0)
                    self.label = torch.cat([self.label,torch.tensor([i]*(len(buf[key])))],dim=0)
        # self.items = torch.clamp(self.items.reshape(-1,1,32,32),min=-1,max=1)
        self.items = (self.items.reshape(-1,1,32,32)/ 255).to(torch.float32)
        self.label = self.label.to(torch.long)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]


class QUERRY_DATASET(Dataset):
    def __init__(self, querry_data, querry_label):
        self.items = querry_data.to(torch.float32)
        self.label = querry_label.to(torch.long)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx, :], self.label[idx]



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # output: (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (16, 8, 8)
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # output: (32, 4, 4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (32, 2, 2)
            nn.Flatten(),  # output: (128)
            nn.Linear(128, 72),  # final encoding of size 72
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Unflatten(1, (32, 2, 2)),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # output: (32, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # output: (16, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # output: (16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # output: (1, 32, 32)
            nn.Sigmoid()  # Assuming the input pixels were normalized to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

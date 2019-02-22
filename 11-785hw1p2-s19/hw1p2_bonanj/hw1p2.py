
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import time

class WSJ():
    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_net = None
    
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['/Users/bonanjin/Documents/cmu/course/11785/assignment1/part2/11-785hw1p2-s19'], 'dev')
            return self.dev_set

    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['/Users/bonanjin/Documents/cmu/course/11785/assignment1/part2/11-785hw1p2-s19'], 'train')
            return self.train_set
    
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['/Users/bonanjin/Documents/cmu/course/11785/assignment1/part2/11-785hw1p2-s19'], 'test.npy'), encoding='bytes'), None)
        return self.test_set

    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes'))
     
class MyDataset1(Dataset):
    def __init__(self):
        train_data,train_label = load_raw('./','train')
        y,y_label=[],[]
        k = 17
        zeros = np.zeros((k,40))
        for i in range(len(train_data)):
            train_data[i] = np.append(zeros,train_data[i])
            train_data[i] = np.append(train_data[i],zeros).reshape(-1,40)
            for j in range(k,len(train_data[i])-k):
                y.append(train_data[i][j-k:j+k+1])
                y_label.append(train_label[i][j-k])
        y_label = np.array(y_label)
        self.Data = y
        self.Label = y_label
    def __getitem__(self, index):
        sample=torch.Tensor(self.Data[index]).view(-1)
        label=self.Label[index] 
        return sample, label        
    def __len__(self):
        return len(self.Data)
    
class MyDataset2(Dataset):
    def __init__(self):
        valid_data,valid_label = load_raw('./','dev')
        y,y_label=[],[]
        k=17
        zeros = np.zeros((k,40))
        for i in range(len(valid_data)):
            valid_data[i] = np.append(zeros,valid_data[i])
            valid_data[i] = np.append(valid_data[i],zeros).reshape(-1,40)
            for j in range(k,len(valid_data[i])-k):
                y.append(valid_data[i][j-k:j+k+1])
                y_label.append(valid_label[i][j-k])
        #y = np.array(y)
        y_label = np.array(y_label)
        self.Data = y
        self.Label = y_label
    def __getitem__(self, index):
        sample=torch.Tensor(self.Data[index]).view(-1)
        label=self.Label[index]
        return sample,label
    def __len__(self):
        return len(self.Data)


class MyMLP(nn.Module):
    def __init__(self, size_list):
        super(MyMLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
            nn.BatchNorm1d(size_list[i+1])
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.net(x)
    
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(device)

    running_loss = 0.0
    
    start_time = time.time()
    for i, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   
        data = data.to(device)
        target = target.long().to(device)

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        if i%9000 == 0:
            print ("Processing..." , i/9000)
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for i, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.long().to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print (device)
train_Dataset = MyDataset1()
test_Dataset = MyDataset2()
dataloader_args = dict(shuffle=True, batch_size=256, pin_memory=True) if cuda                         else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(dataset=train_Dataset, **dataloader_args) 
test_loader = DataLoader(dataset=test_Dataset, **dataloader_args)
model = MyMLP([1240,2048,2048,1024,1024,512,256,138])
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.001)
#n_epochs = 50
Train_loss = []
Test_loss = []
Test_acc = []
learning_r = 0.001
for i in range(0,20):
    learning_r = learning_r * 0.5
    print('epoch: ',  i+1)
    optimizer = optim.Adam(model.parameters(),lr=learning_r)
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model, test_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)

import numpy as np
testdata = np.load(os.path.join('./', '{}.npy'.format('test')), encoding='bytes')

class MyDataset3(Dataset):
    def __init__(self):
        testdata = np.load(os.path.join('./', '{}.npy'.format('test')), encoding='bytes')
        y=[]
        k=17
        zeros = np.zeros((k,40))
        for i in range(len(testdata)):
            testdata[i] = np.append(zeros,testdata[i])
            testdata[i] = np.append(testdata[i],zeros).reshape(-1,40)
            for j in range(k,len(testdata[i])-k):
                y.append(testdata[i][j-k:j+k+1])
        self.Data = y
    
    def __getitem__(self, index):
        sample = torch.Tensor(self.Data[index]).view(-1)
        return sample
    
    def __len__(self):
        return len(self.Data)
testloader_args = dict(shuffle=False, batch_size=256, pin_memory=True) if cuda                         else dict(shuffle=False, batch_size=64)
result_Dataset = MyDataset3()
result_loader = DataLoader(dataset=result_Dataset, **testloader_args)
result=[]
for i, data in enumerate(result_loader):
    data = data.to(device)
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    result.append(predicted)

import csv
import codecs
res=[]
for i in range(len(result)):
    for j in range(len(result[i])):
        res.append(result[i][j].item())
with open("./test.csv","w", newline="") as csvfile:
    #csvfile.write(codecs.BOM_UTF8)
    writer = csv.writer(csvfile,dialect=("excel"))  
    writer.writerow(['id','label'])
    for i in range(len(res)):
        writer.writerow([i,res[i]])


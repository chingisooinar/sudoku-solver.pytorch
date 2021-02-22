#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:23:10 2021

@author: nuvilabs
"""
# import the usual resources
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from trainer import train_net
from validater import val_net
import argparse
from model import SudokuCNN
from data_preprocess import get_data
from utils import DatasetInstance
from accuracy import *
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch",default=2, type=int, help="number of epochs")
ap.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
ap.add_argument("-m", "--model", default="CNN", type=str, help="Model Name")
ap.add_argument( "--test", default=True, type=bool, help="Test mode")
ap.add_argument("-r", "--resume", default="CNN_0.11553671139955521.pt", type=str, help="checkpoint")
args = vars(ap.parse_args())
ROOT = None
if args['model'] == "CNN":
    net = SudokuCNN()
    if args['resume']:
        net.load_state_dict(torch.load('./saved_models/' + args['resume']))
model_name = args['model']
print(net)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = get_data('sudoku.csv')
#convert to tensors
X_train = torch.tensor(x_train).type(torch.FloatTensor)
X_test = torch.tensor(x_test).type(torch.FloatTensor)
y_train = torch.tensor(y_train).type(torch.LongTensor)
y_test = torch.tensor(y_test).type(torch.LongTensor)

train_dataset = DatasetInstance(X_train, y_train)
test_dataset = DatasetInstance(X_test, y_test)
# load training data in batches
batch_size = args["batch_size"]
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)
# load test data in batches
batch_size =args["batch_size"]

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=0)


criterion = nn.CrossEntropyLoss().cuda()

net.to(device)

optimizer =torch.optim.Adam(net.parameters(), lr=1e-4) #define optimizer
#val_net(0, 0, net, criterion, optimizer, device, train_loader)
if args['test']:
    test_accuracy(0, 0, net, device, test_loader)
else:
    n_epochs = args["epoch"]
    min_loss = float('inf')
    for epoch in range(n_epochs):
        print("=====Training=======")
        train_net(epoch, n_epochs, net, criterion, optimizer, device, train_loader)
        print("=====Validation=======")
        loss = val_net(epoch, n_epochs, net, criterion, optimizer, device, test_loader)
        if loss < min_loss:
            print("=====Saving=======")
            model_dir = './saved_models/'
            name =  model_name+'_'+str(loss)+'.pt'
            min_loss = loss
            # after training, save your model parameters in the dir 'saved_models'
            torch.save(net.state_dict(), model_dir+name)

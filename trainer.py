#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:57:08 2021

@author: nuvilabs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def train_net(epoch, n_epochs, net, criterion, optimizer, device, train_loader):

    # prepare the net for training
    net.train()

    print(device)
    running_loss = 0.0

    # train on batches of data, assumes you already have train_loader
    for batch_i, data in enumerate(train_loader):
        # get the input  and their corresponding targets
        sudoku = data[0]
        target = data[1]
        # zero the parameter (weight) gradients
        optimizer.zero_grad()
        
        sudoku = sudoku.to(device)
        target = target.to(device)
        sudoku = Variable(sudoku)
        target = Variable(target)
        # forward pass to get outputs
        output = net(sudoku)

        # calculate the loss between predicted and target 
        loss = criterion(output , target)
        
        # backward pass to calculate the weight gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # print loss statistics
        running_loss += loss.item()
        if batch_i % 200 == 0:    # print every 10 batches
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
            running_loss = 0.0

    #print('Finished Training')

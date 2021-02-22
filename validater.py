#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:02:16 2021

@author: nuvilabs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def val_net(epoch, n_epochs, net, criterion, optimizer, device, test_loader):
    net.eval()      
    val_loss=0.0
    print(device)
    correct = 0
    ndata = 0
    with torch.set_grad_enabled(False):
        for i,data in enumerate(test_loader):
            sudoku = data[0]
            target = data[1]
            optimizer.zero_grad()
            sudoku = sudoku.to(device)
            target = target.to(device)
            # forward pass to get outputs
            output = net(sudoku)
    
            # calculate the loss between predicted and target            
            loss = criterion(output , target)
    
            val_loss+=loss.data.item()
            preds = torch.argmax(output,dim=1)
           
            for j in range(len(preds)):
                if (preds[j] == target[j]).all():
                    correct += 1
            ndata += preds.shape[0]
            if i==0 or i % 1000 == 0:
                print(f'Test Loss at {i}: {val_loss / (i + 1)} Accuracy:{correct}/{ndata}')
    print(f'Test Loss at {i}: {val_loss / (i + 1)} Accuracy:{correct/ndata}')
    return val_loss / (i + 1)

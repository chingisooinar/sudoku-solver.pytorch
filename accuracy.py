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
def norm(a):
    
    return (a/9)-.5
def denorm(a):
    
    return (a+.5)*9

def inference_sudoku(sample,model):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = sample.clone().cuda()
    
    while True:
    
        out = model(feat.reshape((1,1,9,9)))  
        
        pred = torch.argmax(out, axis=1).reshape((9,9)) + 1
        prob,_ = torch.max(out, axis=1)
        #prob = prob.reshape((9,9))
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
        prob = torch.sigmoid(prob)   
        prob_new = prob.flatten() * mask.flatten()
    
        ind = torch.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    return pred 

def test_accuracy(epoch, n_epochs, net, device, test_loader):
    net.eval()      
    print(device)
    correct = 0
    ndata = 0
    with torch.set_grad_enabled(False):
        for i,data in enumerate(test_loader):
            sudoku = data[0]
            target = data[1]
            sudoku = sudoku.to(device)
            target = target.to(device)
            
           
            for j in range(len(sudoku)):
                if (inference_sudoku(sudoku[j],net) == target[j] + 1).all():
                    correct += 1
                
            ndata += sudoku.shape[0]
            if ndata>=1000:
                print(f' Accuracy:{correct}/{ndata}')
    print(f'Final Accuracy:{correct/ndata}')
    return val_loss / (i + 1)

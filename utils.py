#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:55:06 2021

@author: nuvilabs
"""
from torch.utils.data import Dataset
import numpy as np
class DatasetInstance(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X :sudoku, y: solutions

        """
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index):
        
        sudoku = self.X[index]
        target = self.y[index]
        return sudoku,target
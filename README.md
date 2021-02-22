# How to Solve Sudoku with Convolutional Neural Networks (CNN)

Using Relational Reasoning in Deep Learning to Solve Sudoku

# About

Logic-based games, including Sudoku, were shown to help delay neurological disorders like Alzheimer and dementia. Sudoku is a 9 by 9 grid where people need to fill it in with digits so that each column, each row, and each of the nine 3×3 subgrids that make up the board contain all of the digits from 1 to 9. I explore the use of Convolutional Neural Networks in this old yet lovely game. I would like to acknowledge that this article is inspired by this lovely work, which is presented as a project for Stanford's Deep Learning course.

Link: [https://cs230.stanford.edu/files_winter_2018/projects/6939771.pdf](https://cs230.stanford.edu/files_winter_2018/projects/6939771.pdf)

# Dataset

Before I start discussing my solution I would like to note that the dataset I am using might be different. Additionally, my evaluation metric is Number of solved sudoku / Total Number of sudoku, whereas the authors calculate Number of blanks where the prediction matched the solution / Number of blanks. Therefore, it is hard to compare my test results. The dataset called "1 Million Sudoku Games" is publically available online on Kaggle. The architecture was trained on 800,000 sudoku games and validated on 200,000 games.

Link: [https://www.kaggle.com/bryanpark/sudoku](https://www.kaggle.com/bryanpark/sudoku)

# Data processing

Considering that we are feeding unsolved sudoku into CNN, we reshape it to have a shape of (1, 9, 9). Additionally, we normalize it by dividing it by 9 and subtracting 0.5 due to which we achieve zero mean-centred data. Neural networks commonly perform much better with zero centred normalized data. 

# Model Architecture

I am using a very similar architecture that is presented in the work. So, the authors propose a 16 layer CNN where each of the first 15 layers having 512 filters, with the final layer being a 1 by 1 convolution with 9 filters. The model can be viewed in model.py

# Training

I trained the network for 5 epochs, with batch size 64. The learning rate was chosen to be 1e-4. Finally, I used Adam optimizer, which is mentioned by the authors, and CrossEntropy loss. The final validation loss ended up being about 0.11.

# Results

CNNs can be efficiently used to solve Sudoku games, my test accuracy is approximately 93%.
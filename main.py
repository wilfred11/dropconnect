import math

import scipy.stats
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm
import torchvision.transforms as transforms

import stat_test
from model import check_accuracy, NN, NN_dropconnect
import numpy as np
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import norm
import seaborn as sns
from statistics import NormalDist
import pickle

do = 11

input_size = 784  # 28x28 pixels
num_classes = 10  # digits 0-9
learning_rate = 0.001
batch_size = 64
num_epochs = 100

if do==1:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
    K = len(train_dataset)  # enter your length here
    subsample_train_indices = torch.randperm(len(train_dataset))[:K]
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(subsample_train_indices))

    test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
    K_test = len(test_dataset)  # enter your length here
    subsample_test_indices = torch.randperm(len(test_dataset))[:K_test]
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             sampler=SubsetRandomSampler(subsample_test_indices))
    model = NN_dropconnect(input_size=input_size, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_list=[]
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            # Reshape data to (batch_size, input_size)
            data = data.reshape(data.shape[0], -1)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)
            loss_list.append(loss)
            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()
    # Final accuracy check on training and test sets
    train_num_correct, train_num_samples, train_accuracy=check_accuracy(train_loader, model, device)
    test_num_correct, test_num_samples, test_accuracy =check_accuracy(test_loader, model, device)
    results =[[train_num_samples, train_accuracy/100], [test_num_samples, test_accuracy/100]]
    with open('result_drop.pkl', 'wb') as f:
        pickle.dump(results, f)

    stat_test.test(train_num_samples, test_num_samples, train_accuracy/100, test_accuracy/100)

if do == 11:
    with open('result.pkl', 'rb') as f:
        no_drop = pickle.load(f)
    with open('result_drop.pkl', 'rb') as f:
        drop = pickle.load(f)
    print(no_drop)
    print(drop)
    stat_test.test(no_drop[1][0], drop[1][0], no_drop[1][1], drop[1][1])
    stat_test.show_binoms_as_normal_approx(no_drop[1][0], drop[1][0], no_drop[1][1], drop[1][1])












































































































































































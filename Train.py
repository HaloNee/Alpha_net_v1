import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

import torch
    
from torch.utils.data import Dataset, DataLoader

from alpha_net_model import *

#定义训练函数

def cal_cls_accuracy(y_pred, y):
    y_pred = y_pred.argmax(dim = 1)
    cond = y_pred == y
    accuracy = sum(cond) / y_pred.shape[0]
    return(accuracy)

def cal_accuracy(y_pred, y):
    cond = abs(y_pred - y) < 1e-1
    accuracy = sum(cond) / y.shape[0]
    return accuracy
class MyData(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]

save_path = f'../data_output/alpha_net_v1_trained'
Path(save_path).mkdir(exist_ok=True)
with open(Path(save_path) / 'training_history.csv', mode='w') as file:
    file.write('epoch,batch_num,trainMSELoss,accuracy\n')
with open(Path(save_path) / 'val_history.csv', mode='w') as file:
    file.write('epoch,stock_list_num,valMSELoss,accuracy\n')
alphanet = AlphaNet()
alphanet.float()
lr = 1E-4
optimizer = torch.optim.RMSprop(
    alphanet.parameters(), lr=lr, weight_decay=3e-4)
loss_func = torch.nn.MSELoss()

train_part_num = 2
val_part_num = 1
epochs = 10
batch_num, train_loss, accu_sum = 0, 0, 0
for epoch in range(epochs):
    
    for train_it in range(1, train_part_num + 1):
        print(f"epoch:{epoch+1}, part{train_it} is training")
        X_train = np.load(
            f'../data_input/train_data_set/X_train_part_{train_it}.npy')
        y_train = np.load(
            f'../data_input/train_data_set/y_train_part_{train_it}.npy')
        X_train = X_train.reshape(len(X_train), 1, -1, 60)
        train_dataset = MyData(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
        for idx, (x, y) in enumerate(train_dataloader):
            if torch.isnan(x).any() or torch.isnan(y).any():
                continue
            alphanet.train()
            predict = alphanet(x.float())
            # predict = torch.squeeze(predict)
            accuracy = cal_accuracy(predict, y)   # 计算正确率
            accu_sum += accuracy
            #!!!predict和y形状不一样
            loss = loss_func(predict.float(), y.float())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新网络
            batch_num += 1

            if batch_num % 5 == 0:
                print(f'The {batch_num}th batch is trained')
                with open(Path(save_path) / 'training_history.csv', 'a') as file:
                    file.write(f'{epoch + 1},{batch_num},{train_loss / 5},{accu_sum / 5}\n')
                train_loss, accu_sum = 0, 0
    
    with torch.no_grad():
        val_batch_num, val_loss, accu_count = 0, 0, 0
        for val_it in range(1, val_part_num + 1):
            print(f"epoch:{epoch+1}, part{val_it} is validating")
            X_val = np.load(
                f'../data_input/train_data_set/X_val_part_{val_it}.npy')
            y_val = np.load(
                f'../data_input/train_data_set/y_val_part_{val_it}.npy')
            X_val = X_val.reshape(len(X_val), 1, -1, 60)
            val_dataset = MyData(X_val, y_val)
            val_dataloader = DataLoader(
                val_dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
            
            for x_test, y_test in val_dataloader:
                if torch.isnan(x_test).any() or torch.isnan(y_test).any():
                    continue
                predict = alphanet(x_test.float())
                loss = loss_func(predict.float(), y_test.float())
                val_loss += loss.item()
                accuracy = cal_accuracy(predict, y_test)
                accu_count += accuracy.item()                
                val_batch_num += 1
        with open(Path(save_path) / 'val_history.csv', 'a') as file:
            file.write(
                f'{epoch + 1},{batch_num},{val_loss / val_batch_num},{accu_count / val_batch_num}\n')

torch.save(alphanet, 'alphanet.pth')

trained_model = torch.load('alphanet.pth')
for name, each in trained_model.named_parameters():
    print(name, each, each.shape)
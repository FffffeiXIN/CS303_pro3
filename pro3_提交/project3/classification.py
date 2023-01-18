import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class model(torch.nn.Module):
    def __init__(self, n_features, n_output):  # 搭建计算图的一些变量的初始化
        super(model, self).__init__()  # 将self（即自身）转换成Net类对应的父类，调用父类的__init__（）来初始化
        self.hidden1 = torch.nn.Linear(n_features, 100)
        self.hidden2 = torch.nn.Linear(100, 50)
        self.hidden3 = torch.nn.Linear(50, 20)
        self.out = torch.nn.Linear(20, n_output)

    def forward(self, x):  # 前向传播，搭建计算图的过程
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    information = torch.load("data.pth")
    data = information["feature"]
    target = information["label"]
    print(data)
    print(target)
    print(np.unique(target.numpy()))

    train_set, test_set, train_target, test_target = train_test_split(data, target, test_size=0.2)
    # 处理labels
    train_label = torch.zeros(len(train_target), 10, dtype=torch.float32)
    test_label = torch.zeros(len(test_target), 10, dtype=torch.float32)
    for i in range(len(train_target)):
        train_label[i][train_target[i]] = 1.
    for i in range(len(test_target)):
        test_label[i][test_target[i]] = 1.

    print(train_label)
    print(test_label)

    # initialize model
    net = model(n_features=256, n_output=10)  # 总共10类

    # learning parameters
    lr = 0.001
    epochs = 2000
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    loss_func = nn.CrossEntropyLoss()  # crossentropy为啥不行？

    # train
    scheduler.step()
    net.train()
    round = 0
    for epoch in range(epochs):
        round += 1
        output = net(train_set)
        loss = loss_func(output, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if round % 10 == 0:
            # model_pred = torch.tensor([item.cpu().detach().numpy() for item in output])
            # idx = np.argmax(model_pred, axis=1)
            idx = torch.max(F.softmax(output, dim=1), 1)[1]
            output_all = []
            targets_all = []
            for index in idx:
                output_all.append(index.item())
            for label in train_target:
                targets_all.append(label.item())  # target:不需要向量，数字即可
            print(f'training loss: {loss}')
            print(classification_report(targets_all, output_all))


    # test
    test_output = net(test_set)
    # model_pred = torch.tensor([item.cpu().detach().numpy() for item in test_output])
    # idx = np.argmax(model_pred, axis=1)
    idx = torch.max(F.softmax(test_output, dim=1), 1)[1]
    output_all = []
    targets_all = []
    for index in idx:
        output_all.append(index.item())
    for label in test_target:
        targets_all.append(label.item())  # target:不需要向量，数字即可
    print(classification_report(targets_all, output_all))

    # torch.save(net, 'model.pth')
    torch.save(net.state_dict(), 'model.pth')

    # 不应该都拿来训练吗？？？

import torch
from typing import Tuple
from .src import RADIUS, N_CTPS, evaluate, compute_traj
import torch.nn.functional as F
import time
import numpy as np
from functorch import vmap
import random


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


class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        # 加载训练好的分类器
        self.model = model(n_features=256, n_output=10)
        self.model.load_state_dict(torch.load('./project3/model.pth'))
        self.model.requires_grad_(False)
        self.model.eval()

    def vmap_function(self, ctps_inter):
        score = evaluate(compute_traj(ctps_inter), self.target_pos, self.target_scores, RADIUS)
        return score

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile.

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 目标物当前的位置
            target_features: features of shape `(N, d)`.目标物的特征
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
            k个class，每个class对应的分数是多少
        Return: Tensor of shape `(N_CTPS-2, 2)` 三个控制点的坐标 3*2的张量
            the second to the second last control points
        """
        start_time = time.time()

        self.target_pos = target_pos

        out = self.model(target_features)
        idx = torch.max(F.softmax(out, dim=1), 1)[1]
        target_scores = class_scores[idx]
        self.target_scores = target_scores

        best_score = torch.tensor(-float('inf'))
        res = np.array([[1, 1000000], [2, 1000000], [3, 1000000]])
        low = torch.from_numpy(res)
        best_pos = torch.from_numpy(res)

        while time.time() - start_time < 0.27:
            ctps_inter = torch.randn((200, N_CTPS - 2, 2)) * torch.tensor([5., 2.]) + torch.tensor([0., -1.])
            # ctps_inter = torch.randn((200, N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            score = vmap(self.vmap_function)(ctps_inter)

            out, idx = torch.max(score, dim=0)
            if score[idx] > best_score:
                best_score = score[idx]
                best_pos = ctps_inter[idx]

        while time.time() - start_time < 0.285:
            print('1')
            ctps_inter = torch.randn((55, N_CTPS - 2, 2)) * torch.tensor([5., 2.]) + torch.tensor([0., -1.])
            # ctps_inter = torch.randn((200, N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            score = vmap(self.vmap_function)(ctps_inter)

            out, idx = torch.max(score, dim=0)
            if score[idx] > best_score:
                best_score = score[idx]
                best_pos = ctps_inter[idx]

        print(time.time()-start_time)
        if best_score < 0:
            return low

        return best_pos

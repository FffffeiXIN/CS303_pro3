import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import random
from typing import Tuple

from functorch import vmap

from .src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj


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


def my_evaluate(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    my_hit = 1. * torch.sigmoid(-40. * (d - 0.3))
    # print(my_hit)
    my_hit.requires_gard = True
    value = torch.sum(my_hit * target_scores, dim=-1)
    return value


def mine_evaluate(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float
) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = d <= radius
    d[hit] = 1
    d[~hit] = radius / d[~hit]
    value = torch.sum(d * target_scores, dim=-1)
    return value


class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        self.net = model(256, 10)
        self.net.load_state_dict(torch.load('./project3/model.pth'), strict=True)
        self.net.requires_grad_(False)
        self.net.eval()

    def vmap_function(self, ctps_inter):
        score = evaluate(compute_traj(ctps_inter), self.target_pos, self.target_scores, RADIUS)
        return score

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        self.target_pos = target_pos
        assert len(target_pos) == len(target_features)

        out = self.net(target_features)
        idx = torch.max(F.softmax(out, dim=1), 1)[1]
        target_scores = class_scores[idx]
        self.target_scores = target_scores

        best_score = torch.tensor(-float('inf'))
        res = np.array([[1, 1000000], [2, 1000000], [3, 1000000]])
        low = torch.from_numpy(res)
        best_pos = torch.from_numpy(res)

        cnt = 0
        while time.time() - start < 0.285:
            cnt += 1
            print('--------------------------' + str(cnt) + '-------------------------------------------')
            ctps_inter = torch.rand((200, N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            # ctps_inter = torch.randn((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            # score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
            score = vmap(self.vmap_function)(ctps_inter)
            out, idx = torch.max(score, dim=0)
            if score[idx] > best_score:
                best_score = score[idx]
                best_pos = ctps_inter[idx].detch().clone()
            # for i in range(20):
            #     temp_points = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            #     temp_score = evaluate(compute_traj(temp_points), target_pos, target_scores, 0.3)
            #     if temp_score > score:
            #         score = temp_score
            #         ctps_inter = temp_points
            ctps_inter = best_pos.detch().clone()
            ctps_inter.requires_grad = True
            lr = 0.4  # 参数可调
            optimizer = optim.Adam([ctps_inter], lr=lr, maximize=True)
            # optimizer = torch.optim.AdamW([ctps_inter], lr=lr, maximize=True, weight_decay=0)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            #                                                        mode='min', factor=0.5, patience=1, verbose=False,
            #                                                        threshold=0.0001, threshold_mode='rel', cooldown=0,
            #                                                        min_lr=0,
            #                                                        eps=1e-08)

            for i in range(12):
                # cnt += 1
                optimizer.zero_grad()
                real_score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
                # gra_score = mine_evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
                gra_score = my_evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
                print(' my score = ' + str(gra_score.item()) + " real score = " + str(
                    real_score.item()) + " best score = " + str(best_score.item()))
                if real_score > best_score:
                    best_score = real_score
                    best_pos = ctps_inter.detch().clone()
                gra_score.backward()
                optimizer.step()
                # scheduler.step(real_score)
                # for param_group in optimizer.param_groups:  #
                #     print(param_group)

            # print('cnt1 = ' + str(cnt))

        if best_score < 0:
            return low
        else:
            return best_pos

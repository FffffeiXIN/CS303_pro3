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
    def __init__(self, n_features, n_output):
        super(model, self).__init__()
        self.hidden1 = torch.nn.Linear(n_features, 100)
        self.hidden2 = torch.nn.Linear(100, 50)
        self.hidden3 = torch.nn.Linear(50, 20)
        self.out = torch.nn.Linear(20, n_output)

    def forward(self, x):
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
    my_hit = 1. * torch.sigmoid(-60. * (d - 0.3))
    # print(my_hit)
    my_hit.requires_gard = True
    value = torch.sum(my_hit * target_scores, dim=-1)
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

        while time.time() - start < 0.25:
            ctps_inter = torch.randn((200, N_CTPS - 2, 2)) * torch.tensor([N_CTPS, 2.]) + torch.tensor([0., -1.])
            score = vmap(self.vmap_function)(ctps_inter)
            out, idx = torch.max(score, dim=0)
            if score[idx] > best_score:
                best_score = score[idx]
                best_pos = ctps_inter[idx].detach().clone()
        ctps_inter = best_pos.detach().clone()
        ctps_inter.requires_grad = True
        lr = 0.05  # 参数可调
        optimizer = optim.Adam([ctps_inter], lr=lr, maximize=True)

        for i in range(12):
            if time.time() - start > 0.285:
                break
            optimizer.zero_grad()
            real_score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
            gra_score = my_evaluate(compute_traj(ctps_inter), target_pos, target_scores, 0.3)
            # print(' my score = ' + str(gra_score.item()) + " real score = " + str(
            #     real_score.item()) + " best score = " + str(best_score.item()))
            if real_score > best_score:
                best_score = real_score
                best_pos = ctps_inter.detach().clone()
            gra_score.backward()
            optimizer.step()

        # print('time = ' + str(time.time() - start))
        if best_score < 0:
            return low
        else:
            return best_pos

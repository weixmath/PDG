import torch
import numpy as np
import torch.nn as nn
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.zeros((B,),dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.F1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.F2 = nn.Sequential(nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(),
                                nn.Linear(256, K * K))

    def forward(self, x, params=None):
        x = self.F1(x)
        x = torch.max(x, 2)[0]
        x = x.view(x.shape[0], -1).contiguous()
        x = self.F2(x)
        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device='cuda')
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def normal_pc(pc):
    pc_mean = torch.mean(pc, 1, keepdim=True)
    pc = pc - pc_mean
    pc_L_max = torch.max(torch.sqrt(torch.sum(pc ** 2, -1)))
    pc = pc / pc_L_max
    return pc


def knn2(x, y, k):
    inner = -2 * torch.matmul(x, y.transpose(1, 2))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=1)[1]  # (batch_size, num_points, k)
    return idx

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx





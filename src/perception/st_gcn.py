"""
ST-GCN 模型定义 + MediaPipe 骨架图结构。

基于 "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (Yan et al., 2018).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== 骨架图 ==========================

class _MediaPipeGraph:
    """MediaPipe Pose 33 关键点骨架图，3 分区归一化邻接矩阵。"""

    def __init__(self):
        self.num_nodes = 33

        self.edges = [
            # 面部
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (0, 9), (0, 10),
            # 上肢
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
            (17, 19), (18, 20),
            # 躯干
            (11, 23), (12, 24), (23, 24),
            # 下肢
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),
            # 面部-肩部连接
            (7, 11), (8, 12),
        ]

        self.adj = np.zeros((33, 33), dtype=np.float32)
        for i, j in self.edges:
            self.adj[i, j] = 1.0
            self.adj[j, i] = 1.0

        self.dist = self._bfs_distance()
        self.A = self._build_partitioned_adj()  # (3, 33, 33)

    def _bfs_distance(self):
        from collections import deque
        dist = np.full(self.num_nodes, np.inf)
        q = deque([23, 24])
        dist[23] = 0
        dist[24] = 0
        while q:
            u = q.popleft()
            for v in range(self.num_nodes):
                if self.adj[u, v] > 0 and np.isinf(dist[v]):
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _build_partitioned_adj(self):
        A = np.zeros((3, self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj[i, j] == 0:
                    continue
                if self.dist[i] == self.dist[j]:
                    A[0, i, j] = 1.0
                elif self.dist[i] > self.dist[j]:
                    A[1, i, j] = 1.0
                else:
                    A[2, i, j] = 1.0
        for i in range(self.num_nodes):
            A[0, i, i] = 1.0
        for k in range(3):
            D = np.sum(A[k], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                D_sqrt_inv = np.diag(np.where(D > 0, 1.0 / np.sqrt(D), 0.0))
            A[k] = D_sqrt_inv @ A[k] @ D_sqrt_inv
        return A


# ========================== 网络层 ==========================

class _SpatialGraphConv(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.K = A.size(0)
        self.register_buffer("A", A)
        self.M = nn.Parameter(torch.ones(self.K, A.size(1), A.size(2)))
        self.conv = nn.ModuleList([nn.Conv2d(in_c, out_c, 1) for _ in range(self.K)])

    def forward(self, x):
        N, C, T, V = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(N * T, V, C)
        out = 0
        for k in range(self.K):
            A_k = self.A[k] * self.M[k]
            x_k = torch.matmul(A_k, x_flat)
            x_k = x_k.view(N, T, V, C).permute(0, 3, 1, 2)
            out += self.conv[k](x_k)
        return out


class _TemporalConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, (kernel_size, 1), (stride, 1), (pad, 0))
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, kernel=9, stride=1, residual=True):
        super().__init__()
        self.sgcn = _SpatialGraphConv(in_c, out_c, A)
        self.bn_s = nn.BatchNorm2d(out_c)
        self.tgcn = _TemporalConv(out_c, out_c, kernel, stride)
        self.bn_t = nn.BatchNorm2d(out_c)
        self.residual = residual
        if not residual:
            self.res_conv = None
        elif in_c != out_c or stride != 1:
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        res = self.res_conv(x) if self.residual else 0
        x = self.sgcn(x)
        x = self.bn_s(x)
        x = F.relu(x)
        x = self.tgcn(x)
        x = self.bn_t(x)
        x = x + res
        return F.relu(x)


# ========================== 主模型 ==========================

class STGCN(nn.Module):
    """
    ST-GCN 骨架动作分类模型（逐帧输出）。

    Input:  (N, 3, T, 33, 1) 或 (N, 3, T, 33)
    Output: (N, num_classes, T)
    """

    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        graph = _MediaPipeGraph()
        A = torch.from_numpy(graph.A).float()
        self.register_buffer("init_A", A)

        self.bn_input = nn.BatchNorm2d(in_channels)

        self.block1 = _STGCNBlock(in_channels, 64, A, residual=False)
        self.block2 = _STGCNBlock(64, 64, A)
        self.block3 = _STGCNBlock(64, 64, A)
        self.block4 = _STGCNBlock(64, 128, A)
        self.block5 = _STGCNBlock(128, 128, A)
        self.block6 = _STGCNBlock(128, 128, A)
        self.block7 = _STGCNBlock(128, 256, A)
        self.block8 = _STGCNBlock(256, 256, A)
        self.block9 = _STGCNBlock(256, 256, A)

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(-1)  # (N, C, T, V, 1) → (N, C, T, V)

        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = x.mean(dim=-1, keepdim=True)
        x = self.classifier(x).squeeze(-1)
        return x

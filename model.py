# 构建GNN模型示例
import torch
import torch_geometric
from torch_geometric.nn import GCNConv

class PoseSimilarityGNN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.dropout = torch.nn.Dropout(p=0.3)  # 新增 Dropout
        self.fc = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)
        
    def forward(self, x1, edge_index1, x2, edge_index2):
        # 处理第一个姿势图
        h1 = self.conv1(x1, edge_index1).relu()
        h1 = self.dropout(h1)  # 在卷积层后使用 Dropout
        h1 = self.conv2(h1, edge_index1).relu()
        h1 = torch.mean(h1, dim=0)  # 图级池化
        
        # 处理第二个姿势图
        h2 = self.conv1(x2, edge_index2).relu()
        h2 = self.dropout(h2)
        h2 = self.conv2(h2, edge_index2).relu()
        h2 = torch.mean(h2, dim=0)  # 图级池化
        
        # 计算相似度
        diff = torch.abs(h1 - h2)
        diff = self.fc(diff).relu()
        similarity = torch.sigmoid(self.out(diff))
        
        return similarity
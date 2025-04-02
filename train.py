import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Noto Sans')
from model import PoseSimilarityGNN
from preprocess import process_pose_pair

# 自定义数据集类
class PosePairDataset(Dataset):
    def __init__(self, poses_data, similarity_labels):
        """
        Args:
            poses_data: 包含姿势对的列表，每一项是(pose1, pose2)，其中每个pose是[18, 3]的数组
            similarity_labels: 相似度标签，值域为[0, 1]
        """
        self.poses_data = poses_data
        self.similarity_labels = similarity_labels
        
    def __len__(self):
        return len(self.similarity_labels)
    
    def __getitem__(self, idx):
        pose1, pose2 = self.poses_data[idx]
        label = self.similarity_labels[idx]
        
        # 处理姿势对，获取图结构
        graph1, graph2 = process_pose_pair(pose1, pose2)
            
        return graph1, graph2, torch.tensor([label], dtype=torch.float)

# 自定义批处理函数
def collate_pose_pairs(batch):
    """处理图对的批处理"""
    graphs1, graphs2, labels = zip(*batch)
    
    # 批处理图
    batched_graph1 = Batch.from_data_list(list(graphs1))
    batched_graph2 = Batch.from_data_list(list(graphs2))
    
    # 堆叠标签
    batched_labels = torch.cat(labels, dim=0)
    
    return batched_graph1, batched_graph2, batched_labels

# 加载数据函数
def load_pose_data(data_path):
    """
    从文件加载姿势数据
    
    Args:
        data_path: 数据文件路径 (JSON或NPZ格式)
    
    Returns:
        pose_pairs: 姿势对列表
        similarity_labels: 相似度标签数组
    """
    pose_pairs = []
    labels = []
    
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            pose1 = np.array(item['pose1'])
            pose2 = np.array(item['pose2'])
            similarity = item['similarity']
            
            pose_pairs.append((pose1, pose2))
            labels.append(similarity)
            
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
        poses1 = data['poses1']
        poses2 = data['poses2']
        similarities = data['similarities']
        
        for i in range(len(similarities)):
            pose_pairs.append((poses1[i], poses2[i]))
            labels.append(similarities[i])
    
    else:
        raise ValueError("不支持的数据格式，请使用.json或.npz文件")
        
    return pose_pairs, np.array(labels)

# 模型训练函数
def train_pose_similarity_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cuda'):
    """
    训练姿势相似度模型
    
    Args:
        model: PoseSimilarityGNN模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 训练设备
    """
    # 确认使用设备
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 记录训练历史
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for graphs1, graphs2, labels in train_loader:
            # 将数据移到指定设备
            graphs1 = graphs1.to(device)
            graphs2 = graphs2.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(graphs1.x, graphs1.edge_index, graphs2.x, graphs2.edge_index)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for graphs1, graphs2, labels in val_loader:
                graphs1 = graphs1.to(device)
                graphs2 = graphs2.to(device)
                labels = labels.to(device)
                
                outputs = model(graphs1.x, graphs1.edge_index, graphs2.x, graphs2.edge_index)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pose_similarity_model.pth')
            print(f'模型已保存，验证损失: {val_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练历史')
    plt.savefig('training_history.png')
    plt.show()
    
    return model, history

# 评估函数
def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    
    true_similarities = []
    pred_similarities = []
    
    with torch.no_grad():
        for graphs1, graphs2, labels in test_loader:
            graphs1 = graphs1.to(device)
            graphs2 = graphs2.to(device)
            
            outputs = model(graphs1.x, graphs1.edge_index, graphs2.x, graphs2.edge_index)
            
            true_similarities.extend(labels.cpu().numpy().flatten())
            pred_similarities.extend(outputs.cpu().numpy().flatten())
    
    # 计算评估指标
    mse = mean_squared_error(true_similarities, pred_similarities)
    r2 = r2_score(true_similarities, pred_similarities)
    
    print(f'测试MSE: {mse:.4f}')
    print(f'测试R²: {r2:.4f}')
    
    # 绘制真实值与预测值对比图
    plt.figure(figsize=(8, 8))
    plt.scatter(true_similarities, pred_similarities, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('true_sim')
    plt.ylabel('pred_sim')
    plt.title('true vs predicted similarity')
    plt.savefig('prediction_scatter.png')
    plt.show()
    
    return mse, r2

# 推理函数
def predict_similarity(model, pose1, pose2, device='cpu'):
    """预测两个姿势的相似度"""
    model = model.to(device)
    model.eval()
    
    # 处理姿势对
    graph1, graph2 = process_pose_pair(pose1, pose2)
    
    # 将图数据移到设备
    graph1.to(device)
    graph2.to(device)
    
    # 预测相似度
    with torch.no_grad():
        similarity = model(graph1.x, graph1.edge_index, graph2.x, graph2.edge_index)
    
    return similarity.item()

# 主函数：运行完整训练流程
def main(data_path, batch_size=32, num_epochs=30, learning_rate=0.001):
    """
    运行完整的训练流程
    
    Args:
        data_path: 数据文件路径
        batch_size: 批大小
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    # 加载数据
    pose_pairs, similarity_labels = load_pose_data(data_path)
    print(f"加载了 {len(pose_pairs)} 对姿势数据")
    
    # 划分数据集
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pose_pairs, similarity_labels, test_size=0.2, random_state=42
    )
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        train_pairs, train_labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {len(train_pairs)} 对, 验证集: {len(val_pairs)} 对, 测试集: {len(test_pairs)} 对")
    
    # 创建数据集
    train_dataset = PosePairDataset(train_pairs, train_labels)
    val_dataset = PosePairDataset(val_pairs, val_labels)
    test_dataset = PosePairDataset(test_pairs, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_pose_pairs, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_pose_pairs, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_pose_pairs, num_workers=4
    )
    
    # 创建模型 - 假定节点特征维度为5 (原始x,y + 相对dx,dy + 距离)
    node_feature_dim = 5
    model = PoseSimilarityGNN(node_feature_dim)
    
    # 训练模型
    print("开始训练模型...")
    # trained_model, history = train_pose_similarity_model(
    #     model, train_loader, val_loader,
    #     num_epochs=num_epochs, lr=learning_rate
    # )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_pose_similarity_model.pth'))
    
    # 评估模型
    print("评估模型性能...")
    evaluate_model(model, test_loader)
    
    return model

if __name__ == "__main__":
    # 使用示例
    # 确保data_path指向您的数据文件
    data_path = "pose_similarity_data.json"  # 或 .npz
    
    # 执行完整训练流程
    trained_model = main(
        data_path=data_path,
        batch_size=1,
        num_epochs=30,
        learning_rate=0.001
    )
    
    print("训练完成！")
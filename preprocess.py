import torch
import numpy as np
from torch_geometric.data import Data

def coco_keypoints_to_graph(keypoints):
    """将COCO格式的关键点转换为PyTorch Geometric图结构
    
    Args:
        keypoints: 形状为[18, 2]或[18, 3]的数组，表示x,y,(visibility)
    """
    # COCO关键点连接定义 (骨架结构)
    # 定义哪些关节之间有连接
    edges = [
        # 躯干
        [5, 6], [5, 11], [6, 12], [11, 12],  # 肩膀和臀部
        [0, 1], [1, 2], [2, 3], [3, 4],      # 头部和手臂
        [0, 5], [0, 6],                      # 头部到肩膀
        [5, 7], [7, 9], [6, 8], [8, 10],     # 手臂
        [11, 13], [13, 15], [12, 14], [14, 16]  # 腿部
    ]
    
    # 创建边索引 (PyTorch Geometric格式，需要两个方向)
    edge_index = []
    for edge in edges:
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])  # 添加反向边
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # 节点特征 (关键点坐标)
    x = torch.tensor(keypoints[:, :2], dtype=torch.float)
    
    # 创建PyTorch Geometric数据对象
    graph = Data(x=x, edge_index=edge_index)
    
    return graph

def normalize_keypoints(keypoints):
    """归一化关键点坐标
    
    Args:
        keypoints: [18, 2]或[18, 3]的数组
    """
    # 复制以避免修改原始数据
    kpts = keypoints.copy()
    
    # 计算有效关键点的中心点(通常使用躯干关键点)
    valid_idxs = [5, 6, 11, 12]  # 肩膀和臀部关键点
    center = np.mean(kpts[valid_idxs, :2], axis=0)
    
    # 中心对齐
    kpts[:, :2] = kpts[:, :2] - center
    
    # 尺度归一化 (使用骨骼长度)
    # 计算躯干高度作为尺度参考
    torso_height = np.linalg.norm(np.mean(kpts[[5, 6]], axis=0) - np.mean(kpts[[11, 12]], axis=0))
    kpts[:, :2] = kpts[:, :2] / torso_height
    
    return kpts

def preprocess_keypoints(keypoints, visibility_threshold=0.5):
    """预处理关键点，处理缺失点并执行数据增强
    
    Args:
        keypoints: [18, 3]的数组，最后一维是可见度
        visibility_threshold: 可见度阈值
    """
    # 复制数据
    kpts = keypoints.copy()
    
    # 处理低可见度/缺失关键点
    if kpts.shape[1] == 3:  # 如果有可见度信息
        mask = kpts[:, 2] < visibility_threshold
        
        # 可以用相邻关键点平均值或统计学习的方式填充
        # 这里简单示例用躯干中心填充
        torso_center = np.mean(kpts[[5, 6, 11, 12], :2], axis=0)
        kpts[mask, :2] = torso_center
    
    # 数据增强: 可以添加微小噪声增强鲁棒性
    noise_scale = 0.02
    noise = np.random.normal(0, noise_scale, kpts[:, :2].shape)
    kpts[:, :2] = kpts[:, :2] + noise
    
    return kpts

def extract_node_features(keypoints):
    """提取节点特征
    
    Args:
        keypoints: 预处理后的关键点 [18, 2]或[18, 3]
    """
    # 基础特征: x, y坐标
    features = keypoints[:, :2].copy()
    
    # 提取相对位置特征 (相对于重心)
    center = np.mean(features, axis=0)
    relative_pos = features - center
    
    # 计算到重心的距离
    dist_to_center = np.linalg.norm(relative_pos, axis=1).reshape(-1, 1)
    
    # 组合特征
    node_features = np.hstack([
        features,           # 原始坐标 [x, y]
        relative_pos,       # 相对坐标 [dx, dy]
        dist_to_center      # 到重心距离 [d]
    ])
    
    return node_features

def extract_edge_features(keypoints, edge_index):
    """提取边特征
    
    Args:
        keypoints: [18, 2]关键点坐标
        edge_index: 边连接索引 [2, num_edges]
    """
    edge_features = []
    
    for i in range(edge_index.shape[1]):
        # 获取连接的两个节点
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # 骨骼向量
        bone_vector = keypoints[dst, :2] - keypoints[src, :2]
        
        # 骨骼长度
        bone_length = np.linalg.norm(bone_vector)
        
        # 骨骼方向 (单位向量)
        if bone_length > 0:
            bone_direction = bone_vector / bone_length
        else:
            bone_direction = np.zeros(2)
        
        # 组合特征
        edge_feat = np.concatenate([
            bone_vector,     # 骨骼向量 [dx, dy]
            [bone_length],   # 骨骼长度 [l]
            bone_direction   # 骨骼方向 [dirx, diry]
        ])
        
        edge_features.append(edge_feat)
    
    return np.array(edge_features)

def compute_joint_angles(keypoints, skeleton_tree):
    """计算关节角度
    
    Args:
        keypoints: [18, 2]关键点坐标
        skeleton_tree: 骨架树结构，定义关节层次
    """
    angles = []
    
    # 示例骨架树结构
    # 例如: {0: [1], 1: [2], 2: [3], ...} 表示节点0连接到1，1连接到2等
    
    for parent, children in skeleton_tree.items():
        if len(children) >= 2:  # 有分支的关节
            vectors = []
            for child in children:
                # 计算从父关节到子关节的向量
                v = keypoints[child, :2] - keypoints[parent, :2]
                if np.linalg.norm(v) > 0:
                    v = v / np.linalg.norm(v)  # 归一化
                vectors.append(v)
            
            # 计算夹角 (使用点积)
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    cos_angle = np.dot(vectors[i], vectors[j])
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angles.append(angle)
    
    return np.array(angles)

def extract_geometric_features(keypoints):
    """提取几何特征
    
    Args:
        keypoints: [18, 2]关键点坐标
    """
    # 左右对称性
    left_idxs = [5, 7, 9, 11, 13, 15]    # 左侧关键点索引
    right_idxs = [6, 8, 10, 12, 14, 16]  # 右侧关键点索引
    
    # 计算左右侧的差异
    left_points = keypoints[left_idxs, :2]
    right_points = keypoints[right_idxs, :2]
    # 翻转右侧点的x坐标
    right_points_flipped = right_points.copy()
    right_points_flipped[:, 0] = -right_points_flipped[:, 0]
    
    # 计算对称性指标 (左右点云的平均距离)
    symmetry_score = np.mean(np.linalg.norm(left_points - right_points_flipped, axis=1))
    
    # 姿势复杂度: 关键点分布的空间方差
    pose_variance = np.var(keypoints[:, :2], axis=0).sum()
    
    # 肢体伸展度: 手脚端点到躯干中心的平均距离
    extremity_idxs = [4, 7, 10, 13, 16]  # 手、脚端点
    torso_center = np.mean(keypoints[[5, 6, 11, 12], :2], axis=0)
    extension = np.mean([np.linalg.norm(keypoints[idx, :2] - torso_center) for idx in extremity_idxs])
    
    return np.array([symmetry_score, pose_variance, extension])

def process_pose_pair(pose1, pose2):
    """处理两个姿势并准备用于图神经网络的特征
    
    Args:
        pose1, pose2: 形状为[18, 3]的COCO格式关键点
    """
    # 预处理
    pose1_clean = preprocess_keypoints(pose1)
    pose2_clean = preprocess_keypoints(pose2)
    
    pose1_norm = normalize_keypoints(pose1_clean)
    pose2_norm = normalize_keypoints(pose2_clean)
    
    # 构建图结构
    graph1 = coco_keypoints_to_graph(pose1_norm)
    graph2 = coco_keypoints_to_graph(pose2_norm)
    
    # 增强节点特征
    node_features1 = extract_node_features(pose1_norm)
    node_features2 = extract_node_features(pose2_norm)
    
    graph1.x = torch.tensor(node_features1, dtype=torch.float)
    graph2.x = torch.tensor(node_features2, dtype=torch.float)
    
    # 添加边特征
    edge_features1 = extract_edge_features(pose1_norm, graph1.edge_index.numpy())
    edge_features2 = extract_edge_features(pose2_norm, graph2.edge_index.numpy())
    
    graph1.edge_attr = torch.tensor(edge_features1, dtype=torch.float)
    graph2.edge_attr = torch.tensor(edge_features2, dtype=torch.float)
    
    return graph1, graph2

if __name__ == "__main__":
    # 示例用法
    dataset_dir = "dataset"  # 包含多个动作文件夹的目录
    output_path = "pose_similarity_data.json"  # 输出文件路径

    # 生成训练数据
    preprocess_dataset(dataset_dir, output_path, format='json')
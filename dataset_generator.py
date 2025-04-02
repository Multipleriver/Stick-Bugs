import os
import json
import numpy as np
import cv2
import torch
import random
from pathlib import Path
from tqdm import tqdm

def extract_keypoints_from_image(image_path, model):
    """
    使用YOLOv11-Pose模型从图像中提取人体关键点

    Args:
        image_path: 图像文件路径
        model: 加载好的YOLOv11-Pose模型

    Returns:
        keypoints: [17, 3]的数组，COCO格式关键点 (x, y, confidence)
        或者当未检测到人体时返回None
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 使用YOLOv11-Pose进行推理
    results = model(img)

    # 提取人体关键点(选择置信度最高的人体)
    if len(results) > 0 and hasattr(results[0], 'keypoints'):
        keypoints = results[0].keypoints.data
        if len(keypoints) > 0:
            # 选择置信度最高的人体
            best_person = keypoints[0].cpu().numpy()
            return best_person  # 返回关键点格式为[17, 3]

    print(f"未在图像中检测到人体: {image_path}")
    return None

def create_pose_pairs(keypoints_by_action, num_positive_pairs=5, num_negative_pairs=10):
    """
    为每个动作创建正样本对(同一动作)和负样本对(不同动作)

    Args:
        keypoints_by_action: 字典，{动作名: [关键点数组]}
        num_positive_pairs: 每个动作内生成的正样本对数量
        num_negative_pairs: 每个动作与其他动作生成的负样本对数量

    Returns:
        pose_pairs: 姿势对列表 [(pose1, pose2), ...]
        similarity_labels: 相似度标签列表 [1.0, 0.2, ...]
    """
    pose_pairs = []
    similarity_labels = []
    actions = list(keypoints_by_action.keys())

    # 为每个动作创建正样本对(同一动作内的关键点对)
    for action in actions:
        keypoints = keypoints_by_action[action]
        if len(keypoints) < 2:
            continue

        # 在同一动作内创建正样本对
        pairs_count = min(num_positive_pairs, len(keypoints) * (len(keypoints) - 1) // 2)
        for _ in range(pairs_count):
            idx1, idx2 = random.sample(range(len(keypoints)), 2)
            pose_pairs.append((keypoints[idx1], keypoints[idx2]))
            # 同一动作内相似度设为0.8-1.0之间的随机值
            similarity_labels.append(random.uniform(0.95, 1.0))

    # 创建负样本对(不同动作之间的关键点对)
    for i, action1 in enumerate(actions):
        keypoints1 = keypoints_by_action[action1]
        if len(keypoints1) == 0:
            continue

        for action2 in actions[i+1:]:
            keypoints2 = keypoints_by_action[action2]
            if len(keypoints2) == 0:
                continue

            # 在不同动作间创建负样本对
            pairs_count = min(num_negative_pairs, len(keypoints1) * len(keypoints2))
            for _ in range(pairs_count):
                idx1 = random.randint(0, len(keypoints1) - 1)
                idx2 = random.randint(0, len(keypoints2) - 1)
                pose_pairs.append((keypoints1[idx1], keypoints2[idx2]))
                # 不同动作间相似度设为0.0-0.2之间的随机值
                similarity_labels.append(random.uniform(0.0, 0.05))

    return pose_pairs, similarity_labels

def generate_training_data(dataset_dir, output_path, format='json'):
    """
    从数据集目录生成训练数据文件

    Args:
        dataset_dir: 数据集目录路径
        output_path: 输出文件路径
        format: 输出格式，'json'或'npz'
    """
    print("正在加载YOLOv11-Pose模型...")
    # 加载YOLOv11-Pose模型
    try:
        model = torch.hub.load('ultralytics/yolov11', 'yolov11-pose', pretrained=True)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用本地YOLOv11-Pose模型...")
        from ultralytics import YOLO
        model = YOLO('yolo11n-pose.pt')  # 可能需要先下载模型

    # 遍历数据集目录
    keypoints_by_action = {}
    print("开始处理图像...")

    action_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for action_dir in tqdm(action_dirs, desc="处理动作文件夹"):
        action_path = os.path.join(dataset_dir, action_dir)
        image_files = [f for f in os.listdir(action_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        keypoints_list = []
        for img_file in image_files:
            img_path = os.path.join(action_path, img_file)
            keypoints = extract_keypoints_from_image(img_path, model)
            if keypoints is not None:
                keypoints_list.append(keypoints)

        if keypoints_list:
            keypoints_by_action[action_dir] = keypoints_list
            print(f"从'{action_dir}'中提取了{len(keypoints_list)}个有效姿势")
        else:
            print(f"警告: 未能从'{action_dir}'中提取到有效关键点")

    # 创建姿势对和标签
    print("生成姿势对和相似度标签...")
    pose_pairs, similarity_labels = create_pose_pairs(keypoints_by_action)

    print(f"共生成{len(pose_pairs)}对姿势数据")

    # 保存数据
    if format.lower() == 'json':
        data = []
        for (pose1, pose2), similarity in zip(pose_pairs, similarity_labels):
            data.append({
                'pose1': pose1.tolist(),
                'pose2': pose2.tolist(),
                'similarity': similarity
            })

        with open(output_path, 'w') as f:
            json.dump(data, f)
        print(f"已将数据保存为JSON格式: {output_path}")

    elif format.lower() == 'npz':
        poses1 = np.array([pair[0] for pair in pose_pairs])
        poses2 = np.array([pair[1] for pair in pose_pairs])
        similarities = np.array(similarity_labels)

        np.savez(output_path, poses1=poses1, poses2=poses2, similarities=similarities)
        print(f"已将数据保存为NPZ格式: {output_path}")

    else:
        raise ValueError("不支持的输出格式，请使用'json'或'npz'")

    return output_path

def preprocess_dataset(dataset_dir, output_path='pose_similarity_data.json', format='json'):
    """
    主函数：处理数据集并生成训练数据

    Args:
        dataset_dir: 数据集目录路径
        output_path: 输出文件路径
        format: 输出格式，'json'或'npz'
    """
    print(f"开始处理数据集: {dataset_dir}")

    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        raise ValueError(f"数据集目录不存在: {dataset_dir}")

    # 检查是否有动作文件夹
    action_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not action_dirs:
        raise ValueError(f"数据集目录中未找到动作文件夹")

    print(f"发现{len(action_dirs)}个动作文件夹: {', '.join(action_dirs)}")

    # 生成训练数据
    output_file = generate_training_data(dataset_dir, output_path, format)

    print(f"数据预处理完成，输出文件: {output_file}")
    return output_file
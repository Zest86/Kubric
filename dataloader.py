# -*- coding:utf-8 -*-
"""
@Time: 2024/9/26 16:20
@Auth: Rui Wang
@File: dataloader.py
"""
import os
import torch
from torchvision import transforms
from PIL import Image
import json


class PhysicsPropertyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, indices=None, transform=None):
        """
        Args:
            root_dir (string): 存储视频文件夹的根目录路径
            transform (callable, optional): 可选的转换操作
        """
        self.root_dir = root_dir

        self.transform = transform

        # 获取所有视频文件夹的路径
        self.video_folders = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('obj_')
        ]
        self.num_videos = len(self.video_folders)

        # 如果提供了索引范围，则根据索引过滤
        if indices is not None:
            self.video_folders = [self.video_folders[i] for i in indices]

        self.num_videos = len(self.video_folders)

    def __len__(self):
        # 返回数据集的大小（视频文件夹的数量）
        return self.num_videos

    def __getitem__(self, idx):

        transform = transforms.Compose([
            transforms.Resize(256),  # 首先调整图像大小
            transforms.CenterCrop(224),  # 中心裁剪到224x224
            transforms.ToTensor(),  # 转换为Tensor并归一化到[0, 1]
            transforms.Normalize(  # 标准化
                mean=[0.485, 0.456, 0.406],  # ImageNet的均值
                std=[0.229, 0.224, 0.225]  # ImageNet的标准差
            ),
        ])

        # 根据索引获取视频文件夹
        video_folder = self.video_folders[idx]

        # 获取该视频文件夹中的所有图像帧
        image_files = sorted(
            [f for f in os.listdir(video_folder) if f.endswith('.png') and f.startswith('rgba_')],
       )  # 确保帧的顺序
        images = []

        # 读取图像
        for image_file in image_files:
            image_path = os.path.join(video_folder, image_file)
            image = Image.open(image_path).convert('RGB')

            # 可选的图像转换
            if self.transform:
                image = transform(image)

            images.append(image)

        # 读取标注文件
        positions_file = os.path.join(video_folder, 'object_positions.txt')
        object_positions, physical_properties = self.read_positions(positions_file)

        # 返回所有帧的张量和对应的标注信息
        # 有些生成的有问题，需要在这里检查一下，是否为空，为空的话提示一下，需要重做一下数据
        if len(images) == 0:
            print(f"Empty images for {video_folder}")
            return None, None
        return torch.stack(images), object_positions, physical_properties  # shape: (120, C, H, W)

    def read_positions(self, file_path):
        """读取标注文件并解析物体位置和物理属性"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        frame_positions = []
        physical_properties = []

        for line in lines:
            if "Frame" in line:
                # 解析每一帧的物体信息
                frame_info = self.parse_frame_info(line)
                frame_positions.append(frame_info)
            else:
                # 解析物理属性信息
                physical_properties.append(self.parse_physical_properties(line))

        return frame_positions, physical_properties

    def parse_frame_info(self, line):
        """解析每帧的物体信息"""
        parts = line.split(' - ')
        position = self.extract_vector(parts[1])
        orientation = self.extract_quaternion(parts[1])
        object_name = parts[0].split('Frame ')[1].split('Object')[1][1:]

        return {
            'object': object_name,
            'position': position,
            'orientation': orientation
        }

    def extract_vector(self, vector_str):
        """提取位置信息"""
        position_str = vector_str.split('<Vector ')[1].split('>')[0]  # 获取 "<Vector (x, y, z)"
        position = list(map(float, position_str.strip('()').split(',')))
        return position

    def extract_quaternion(self, quaternion_str):
        """提取四元数信息"""
        quaternion_str = quaternion_str.split('<Quaternion ')[1].strip('>').strip()
        cleaned_str = quaternion_str.rstrip('>')
        values = list(map(float, cleaned_str.strip('()').replace('w=', '').replace('x=', '').replace('y=', '').replace('z=', '').split(',')))
        return values

    def parse_physical_properties(self, line):
        """解析物理属性信息"""
        parts = line.split(' - ')
        properties = {}
        object_name = parts[0].split('Object ')[1]
        # 提取质量、摩擦、恢复力和电荷
        for prop in parts[1].split(', '):
            key, value = prop.split('=')
            if key != 'Charge':
                properties[key.strip()] = float(value.strip())
            else:
                # 这里要统一单位
                    magnitude = float(value.strip().split('e')[1])
                    if magnitude == -6:
                        properties[key.strip()] = float(value.strip().split('e')[0])
                    else: # magnitude == -5:
                        properties[key.strip()] = float(value.strip().split('e')[0]) * 10

        return {
            'object': object_name,
            'properties': properties
        }





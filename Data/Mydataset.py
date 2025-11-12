from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import os
from io import BytesIO
import cv2

from torchvision import transforms
from PIL import Image
import os

class Imageset(Dataset):
    def __init__(self, root_dir, is_augment, directory_structure='simple'):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            is_augment (bool): Whether to apply data augmentation.
            directory_structure (str): Directory structure type. Options: 'simple' or 'nested'.
        """
        self.root_dir = root_dir
        self.is_augment = is_augment
        self.directory_structure = directory_structure
        self.data = self._load_data()

        # 定义数据增强和预处理操作
        if self.is_augment:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),  # 调整大小
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomRotation(15),  # 随机旋转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            ])

    def _load_data(self):
        data = []
        if self.directory_structure == 'simple':
            self._load_simple_structure(data)
        elif self.directory_structure == 'nested':
            self._load_nested_structure(data)
        else:
            raise ValueError("Invalid directory_structure. Options: 'simple' or 'nested'.")
        return data

    def _load_simple_structure(self, data):
        for label_dir in os.listdir(self.root_dir):
            label_dir_path = os.path.join(self.root_dir, label_dir)
            if not os.path.isdir(label_dir_path):
                continue

            if label_dir == '0_intact':
                label = 0
            elif label_dir == '1_scratch':
                label = 1
            else:
                print(f"Unknown label directory: {label_dir}. Skipping.")
                continue

            for img_file in os.listdir(label_dir_path):
                img_path = os.path.join(label_dir_path, img_file)
                if not os.path.isfile(img_path):
                    continue

                if not self._is_image_file(img_file):
                    continue

                data.append((img_path, label))

    def _load_nested_structure(self, data):
        for category_dir in os.listdir(self.root_dir):
            category_dir_path = os.path.join(self.root_dir, category_dir)
            if not os.path.isdir(category_dir_path):
                continue

            for label_dir in os.listdir(category_dir_path):
                label_dir_path = os.path.join(category_dir_path, label_dir)
                if not os.path.isdir(label_dir_path):
                    continue

                if label_dir == '0_real':
                    label = 0
                elif label_dir == '1_fake':
                    label = 1
                else:
                    print(f"Unknown label directory: {label_dir} in {category_dir}. Skipping.")
                    continue

                for img_file in os.listdir(label_dir_path):
                    img_path = os.path.join(label_dir_path, img_file)
                    if not os.path.isfile(img_path):
                        continue

                    if not self._is_image_file(img_file):
                        continue

                    data.append((img_path, label))

    def _is_image_file(self, filename):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')  # 使用PIL读取并转换为RGB格式
        img = self.transform(img)  # 应用预处理和数据增强
        return img, label
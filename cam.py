from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torchvision.models as models
import torch
from networks.resnet import resnet50
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import os


# 定义一个简单的数据集类
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_files[idx]


# 加载预训练模型
model = resnet50(num_classes=2)
model.load_state_dict(torch.load('result/resnet/best_model.pth'))
model.eval()

# 选择目标层
target_layer = [model.layer4[-1]]

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
img_dir = '/root/autodl-tmp/sd/patch/0_intact'  # 替换为你的图像文件夹路径
dataset = ImageDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 初始化Grad-CAM
cam = GradCAM(model=model, target_layers=target_layer)

# 创建输出文件夹
output_dir = 'output_cams'
os.makedirs(output_dir, exist_ok=True)

# 处理每个批次的图像
for batch in dataloader:
    input_tensor, img_files = batch
    # 生成CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=None)

    for i, grayscale_cam in enumerate(grayscale_cams):
        # 将热力图叠加到原始图像上
        img_path = os.path.join(img_dir, img_files[i])
        original_image = Image.open(img_path).convert('RGB').resize((224, 224))
        cam_image = show_cam_on_image(np.array(original_image) / 255.0, grayscale_cam, use_rgb=True)

        # 保存结果
        output_path = os.path.join(output_dir, img_files[i])
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

print(f'CAM images saved to {output_dir}')
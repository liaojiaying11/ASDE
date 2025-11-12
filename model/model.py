import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

class SDModel(nn.Module):
    def __init__(self):
        super(SDModel, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # 池化层 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 卷积层 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Dropout 层 1
        self.dropout1 = nn.Dropout(p=0.25)
        # 卷积层 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # 池化层 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Dropout 层 2
        self.dropout2 = nn.Dropout(p=0.25)
        # 卷积层 4
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, stride=8)
        # Flatten 层
        self.flatten = nn.Flatten()
        # Dropout 层 3
        self.dropout3 = nn.Dropout(p=0.5)
        # 全连接层
        self.fc = nn.Linear(6272, 2)

    def forward(self, x):
        # 卷积层 1
        x = self.conv1(x)
        x = F.relu(x)
        # 池化层 1
        x = self.pool1(x)
        # 卷积层 2
        x = self.conv2(x)
        x = F.relu(x)
        # 池化层 2
        x = self.pool2(x)
        # Dropout 层 1
        x = self.dropout1(x)
        # 卷积层 3
        x = self.conv3(x)
        x = F.relu(x)
        # 池化层 3
        x = self.pool3(x)
        # Dropout 层 2
        x = self.dropout2(x)
        # 将卷积层 4 的输入从 4D 转换为 3D
        x = x.view(x.size(0), 1,-1)
        # 卷积层 4
        x = self.conv4(x)
        x = F.relu(x)
        # Flatten 层
        x = self.flatten(x)
        # Dropout 层 3
        x = self.dropout3(x)
        # 全连接层
        x = self.fc(x)
        return x

# 示例用法
if __name__ == "__main__":
    model = SDModel()
    x=torch.randn(1, 3, 112, 112)
    y=model(x)
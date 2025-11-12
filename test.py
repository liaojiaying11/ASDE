from model import SDModel
import torch
import cv2
import numpy as np
import os
from networks.resnet import resnet50
if __name__ == '__main__':
    model = resnet50(num_classes=2)
    model.load_state_dict(torch.load('model_epoch_10.pth'))
    model.eval()
    img = cv2.imread('test.jpg')
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = img.float()
    img = img.unsqueeze(0)

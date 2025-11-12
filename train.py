import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import csv
from networks.resnet import resnet50

from Data import create_dataloader
from model import SDModel


def train(model, device, train_loader, optimizer, criterion, epoch_loss):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss.append(running_loss / len(train_loader))
    print(f'Train Loss: {running_loss / len(train_loader):.4f}')


def validate(model, device, val_loader, criterion, epoch_loss):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    epoch_loss.append(val_loss)
    print(f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment to save results')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存结果的目录
    result_dir = os.path.join('result', args.name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 创建数据加载器
    train_loader = create_dataloader(file_index=r'/root/autodl-tmp/train', batch_size=args.batch_size, num_workers=args.num_workers, is_augument=True, shuffle=True)
    val_loader = create_dataloader(file_index=r'/root/autodl-tmp/val', batch_size=args.batch_size, num_workers=args.num_workers, is_augument=False, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = resnet50(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练参数
    epochs = args.epochs
    best_accuracy = 0.0

    # 用于存储每个 epoch 的损失
    epoch_loss = []

    # 训练循环
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        print(f'Epoch {epoch}/{epochs}')
        train(model, device, train_loader, optimizer, criterion, epoch_loss)
        accuracy = validate(model, device, val_loader, criterion, epoch_loss)

        # 动态调整学习率
        scheduler.step()

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))
            print(f'Saved best model with accuracy: {accuracy:.4f}')

        # 每隔10轮保存一次模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(result_dir, f'model_epoch_{epoch}.pth'))
            print(f'Saved model at epoch {epoch}')

    # 将损失保存到 CSV 文件
    with open(os.path.join(result_dir, 'losses.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        for i in range(0, len(epoch_loss), 2):
            writer.writerow([i // 2 + 1, epoch_loss[i], epoch_loss[i + 1]])
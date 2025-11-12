
from torch.utils.data import DataLoader

import torch
from Data.Mydataset import Imageset


# def create_dataloader(index_file, batch_size, num_workers, is_augument=False,shuffle=True):
#     dataset = ImageLabelDataset(index_file,is_augument)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     return dataloader

def create_dataloader(file_index, batch_size, num_workers, is_augument=False,directory_structure='simple', shuffle=False):
    dataset = Imageset(file_index, is_augument, directory_structure=directory_structure)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_dataloader = create_dataloader(r"D:\Dataset\AIGCBench\test\progan", 8, 0, True, directory_structure="nested")

    total_mean = 0
    total_var = 0
    total_count = 0

    for i, data in enumerate(train_dataloader):
        h, l, _ = data
        h = h.to(device)  # 将数据移动到 GPU

        h = pix_variance(h)  # 计算像素方差

        # 计算当前批次的均值和方差
        batch_mean = torch.mean(h, dim=(0, 2, 3))  # 按批次、高度和宽度计算均值
        batch_var = torch.var(h, dim=(0, 2, 3))  # 按批次、高度和宽度计算方差
        batch_count = h.size(0) * h.size(2) * h.size(3)  # 当前批次的像素总数

        # 累加均值和方差
        total_mean += batch_mean * batch_count
        total_var += batch_var * batch_count
        total_count += batch_count

    # 计算总均值和总方差
    total_mean /= total_count
    total_var /= total_count

    print("总均值:", total_mean.cpu().numpy())  # 将结果移动到 CPU 并转换为 NumPy
    print("总方差:", total_var.cpu().numpy())  # 将结果移动到 CPU 并转换为 NumPy

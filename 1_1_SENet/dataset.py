import numpy as np
import struct
from array import array
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, 
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(self.images[idx], dtype=np.float32).reshape(28, 28)
        label = self.labels[idx]
        
        # 归一化到[0,1]
        image = image / 255.0
        
        # 转换为3通道图像以适配注意力机制模型
        image = np.stack([image, image, image], axis=0)  # (3, 28, 28)
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_mnist_dataloaders(batch_size=32, num_workers=4):
    """
    获取MNIST数据加载器
    """
    import os
    # 设置文件路径（使用项目根目录下的 archive 目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_path = join(project_root, 'archive')
    training_images_filepath = join(archive_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(archive_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(archive_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(archive_path, 't10k-labels.idx1-ubyte')
    
    # 加载数据
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                     test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # 创建数据集
    train_dataset = MNISTDataset(x_train, y_train)
    test_dataset = MNISTDataset(x_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"图像形状: {images.shape}")  # 应该是 (batch_size, 3, 28, 28)
        print(f"标签形状: {labels.shape}")  # 应该是 (batch_size,)
        print(f"图像数据类型: {images.dtype}")
        print(f"标签数据类型: {labels.dtype}")
        break
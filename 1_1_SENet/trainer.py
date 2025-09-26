#!/usr/bin/env python3
"""
SEAttention 模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 添加父目录到路径，以便导入数据集和评估器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_mnist_dataloaders
from model import SEAttention, AttentionClassifier


class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 移除evaluator，因为我们在这里直接实现评估功能
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=5):
        """完整训练过程"""
        print(f"开始训练 SEAttention 模型，共 {epochs} 个epoch")
        print(f"设备: {self.device}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_senet_model.pth')
                print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        return best_val_acc


def main():
    """主函数"""
    print("=" * 60)
    print("SEAttention 模型训练")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载MNIST数据集...")
    # 切换到父目录以正确访问archive文件夹
    original_cwd = os.getcwd()
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.dirname(parent_dir)
    os.chdir(parent_parent_dir)
    
    try:
        train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 创建模型
    print("创建SEAttention模型...")
    attention = SEAttention(input_channels=1, se_channels=64, reduction=16)
    model = AttentionClassifier(attention, num_classes=10)
    
    # 初始化注意力模块权重
    if hasattr(attention, 'init_weights'):
        attention.init_weights()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器并开始训练
    trainer = ModelTrainer(model, device)
    best_acc = trainer.train(train_loader, test_loader, epochs=5)
    
    print(f"\n最终结果:")
    print(f"最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
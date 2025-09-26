#!/usr/bin/env python3
"""
通用的注意力机制模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import Dict, Any, Optional, Tuple


class AttentionClassifier(nn.Module):
    """
    通用的注意力机制分类器
    将注意力模块包装成完整的分类模型
    """
    
    def __init__(self, attention_module: nn.Module, input_channels: int = 3, 
                 num_classes: int = 10, feature_dim: int = 128):
        super(AttentionClassifier, self).__init__()
        
        self.attention_module = attention_module
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 应用注意力机制
        try:
            # 尝试不同的注意力模块调用方式
            if hasattr(self.attention_module, 'forward'):
                # 检查forward方法的参数数量
                import inspect
                sig = inspect.signature(self.attention_module.forward)
                param_count = len([p for p in sig.parameters.values() if p.name != 'self'])
                
                if param_count == 1:
                    # 单输入注意力模块
                    attended_features = self.attention_module(features)
                elif param_count == 2:
                    # 双输入注意力模块（如CrossAttention）
                    attended_features = self.attention_module(features, features)
                elif param_count == 3:
                    # 三输入注意力模块（如SENet_v2）
                    attended_features = self.attention_module(features, features, features)
                else:
                    # 默认单输入
                    attended_features = self.attention_module(features)
            else:
                attended_features = features
                
            # 检查输出形状是否与输入一致
            if attended_features.shape != features.shape:
                attended_features = features
                
        except Exception as e:
            # 如果注意力模块失败，使用原始特征
            print(f"注意力模块执行失败，使用原始特征: {e}")
            attended_features = features
        
        # 分类
        output = self.classifier(attended_features)
        return output


class ModelTrainer:
    """
    通用的模型训练器
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_accuracy = 0.0
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 10, lr: float = 0.001, save_path: Optional[str] = None) -> Dict[str, Any]:
        """完整的训练流程"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"开始训练，共 {epochs} 个epoch")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if save_path:
                    self.save_model(save_path)
        
        total_time = time.time() - start_time
        print(f'训练完成! 总时间: {total_time:.2f}s, 最佳验证准确率: {self.best_accuracy:.2f}%')
        
        return history
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy
        }, path)
        print(f'模型已保存到: {path}')
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f'模型已从 {path} 加载，最佳准确率: {self.best_accuracy:.2f}%')


def create_model_with_attention(attention_module: nn.Module, 
                              input_channels: int = 3, 
                              num_classes: int = 10,
                              feature_dim: int = 128) -> AttentionClassifier:
    """
    创建带有指定注意力机制的分类模型
    """
    return AttentionClassifier(
        attention_module=attention_module,
        input_channels=input_channels,
        num_classes=num_classes,
        feature_dim=feature_dim
    )


if __name__ == "__main__":
    # 测试代码
    from dataset import get_mnist_dataloaders
    
    print("测试训练器...")
    
    # 创建简单的注意力模块进行测试
    class SimpleAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
            
        def forward(self, x):
            return x  # 简单的恒等映射
    
    # 加载数据
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    
    # 创建模型
    attention_module = SimpleAttention(128)
    model = create_model_with_attention(attention_module)
    
    # 创建训练器
    trainer = ModelTrainer(model, device='cpu')
    
    # 训练模型
    history = trainer.train(train_loader, test_loader, epochs=2)
    
    print("训练器测试完成!")
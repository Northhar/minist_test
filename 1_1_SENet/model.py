#!/usr/bin/env python3
"""
SENet (Squeeze-and-Excitation Networks) 模型实现
"""

import numpy as np
import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Networks 注意力机制
    适配版本：先通过卷积层将输入扩展为多通道，然后应用SE注意力机制
    适用于MNIST等单通道输入数据
    """

    def __init__(self, input_channels=1, se_channels=64, reduction=16):
        super().__init__()
        
        # 前置特征提取模块：将输入通道扩展为多通道
        self.feature_extractor = nn.Sequential(
            # 第一层：扩展通道数
            nn.Conv2d(input_channels, se_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(se_channels // 2),
            nn.ReLU(inplace=True),
            
            # 第二层：进一步扩展通道数
            nn.Conv2d(se_channels // 2, se_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(se_channels),
            nn.ReLU(inplace=True),
        )
        
        # SE注意力机制
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(se_channels, se_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels // reduction, se_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.se_channels = se_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: (B, input_channels, H, W) - 对于MNIST是 (B, 1, 28, 28)
        
        # 特征提取：将输入扩展为多通道
        features = self.feature_extractor(x)  # (B, se_channels, H, W)
        
        B, C, H, W = features.size()
        
        # SE注意力机制
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(features).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = features * y
        
        return out


class AttentionClassifier(nn.Module):
    """
    将注意力机制包装成分类器
    适配MNIST数据集：输入为单通道28x28图像
    """
    
    def __init__(self, attention_module, num_classes=10):
        super().__init__()
        self.attention = attention_module
        
        # 获取注意力模块输出的通道数
        se_channels = getattr(attention_module, 'se_channels', 64)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(se_channels, num_classes)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入: (B, 1, 28, 28) for MNIST
        
        # 应用注意力机制
        x = self.attention(x)  # (B, se_channels, 28, 28)
        
        # 分类
        x = self.classifier(x)  # (B, num_classes)
        return x


if __name__ == '__main__':
    print("测试适配MNIST的SEAttention模块:")
    
    # 测试单个SEAttention模块
    print("\n1. 测试SEAttention模块:")
    mnist_input = torch.randn(2, 1, 28, 28)  # MNIST格式输入
    se_attention = SEAttention(input_channels=1, se_channels=64, reduction=16)
    se_output = se_attention(mnist_input)
    print(f"输入形状: {mnist_input.shape}")
    print(f"输出形状: {se_output.shape}")
    
    # 测试不同通道数配置
    print("\n2. 测试不同se_channels配置:")
    se_attention_32 = SEAttention(input_channels=1, se_channels=32, reduction=8)
    se_output_32 = se_attention_32(mnist_input)
    print(f"se_channels=32, 输出形状: {se_output_32.shape}")
    
    # 测试完整分类器
    print("\n3. 测试完整分类器:")
    attention = SEAttention(input_channels=1, se_channels=64, reduction=16)
    classifier = AttentionClassifier(attention, num_classes=10)
    
    # 模拟MNIST批次数据
    batch_input = torch.randn(4, 1, 28, 28)
    batch_output = classifier(batch_input)
    print(f"分类器输入形状: {batch_input.shape}")
    print(f"分类器输出形状: {batch_output.shape}")
    
    # 验证输出维度正确性
    assert batch_output.shape == (4, 10), f"期望输出形状为(4, 10)，实际为{batch_output.shape}"
    
    print("\n✅ 所有测试通过！SEAttention已成功适配MNIST数据集。")
    print("📊 模型架构：")
    print(f"   - 输入：单通道28x28图像")
    print(f"   - 特征提取：1 → {se_attention.se_channels}通道")
    print(f"   - SE注意力：{se_attention.se_channels}通道")
    print(f"   - 分类输出：10类")
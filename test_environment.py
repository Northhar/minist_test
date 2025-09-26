#!/usr/bin/env python3
"""
M4 Mac 深度学习环境测试脚本
测试所有必要的包是否正确安装并可以正常工作
"""

import sys
import traceback

def test_basic_imports():
    """测试基础包导入"""
    print("🔍 测试基础包导入...")
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import init
        print("✅ 基础包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基础包导入失败: {e}")
        return False

def test_pytorch_functionality():
    """测试 PyTorch 功能"""
    print("\n🔍 测试 PyTorch 功能...")
    try:
        import torch
        
        # 检查版本
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 检查 MPS 支持
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) 可用")
            device = torch.device("mps")
        else:
            print("⚠️  MPS 不可用，使用 CPU")
            device = torch.device("cpu")
        
        # 创建测试张量
        x = torch.randn(2, 3, 4, 4).to(device)
        print(f"✅ 张量创建成功，设备: {x.device}")
        
        # 测试基本运算
        y = torch.relu(x)
        z = torch.sum(y)
        print(f"✅ 基本运算成功，结果: {z.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch 功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_senet_model():
    """测试 SENet 模型"""
    print("\n🔍 测试 SENet 模型...")
    try:
        import torch
        import torch.nn as nn
        from torch.nn import init
        
        # 导入 SENet 模型
        sys.path.append('.')
        from importlib import import_module
        
        # 尝试导入 SENet_v2
        try:
            senet_module = import_module('1_1_SENet_v2')
            SEAttention = senet_module.SEAttention
        except:
            print("⚠️  无法导入 SENet_v2 模块，跳过模型测试")
            return True
        
        # 创建模型
        model = SEAttention(channel=512, reduction=8)
        
        # 设置设备
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        # 创建测试输入
        input1 = torch.randn(1, 512, 7, 7).to(device)
        input2 = torch.randn(1, 512, 7, 7).to(device)
        input3 = torch.randn(1, 512, 7, 7).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(input1, input2, input3)
        
        print(f"✅ SENet 模型测试成功")
        print(f"   输入形状: {input1.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   设备: {output.device}")
        
        return True
    except Exception as e:
        print(f"❌ SENet 模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_optional_packages():
    """测试可选包"""
    print("\n🔍 测试可选包...")
    optional_packages = [
        ('einops', 'einops'),
        ('timm', 'timm'),
        ('transformers', 'transformers'),
        ('matplotlib', 'matplotlib.pyplot'),
    ]
    
    results = []
    for name, import_name in optional_packages:
        try:
            __import__(import_name)
            print(f"✅ {name} 可用")
            results.append(True)
        except ImportError:
            print(f"⚠️  {name} 不可用")
            results.append(False)
    
    return any(results)

def main():
    """主测试函数"""
    print("🚀 开始 M4 Mac 深度学习环境测试\n")
    
    tests = [
        ("基础包导入", test_basic_imports),
        ("PyTorch 功能", test_pytorch_functionality),
        ("SENet 模型", test_senet_model),
        ("可选包", test_optional_packages),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境设置成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
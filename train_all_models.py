#!/usr/bin/env python3
"""
自动训练和评估所有注意力机制模型的主脚本
"""

import os
import sys
import glob
import importlib.util
import inspect
import torch
import torch.nn as nn
import json
import traceback
from datetime import datetime
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_mnist_dataloaders
from trainer import ModelTrainer, create_model_with_attention, AttentionClassifier
from evaluator import ModelEvaluator, compare_models


class ModelDiscovery:
    """
    模型发现和加载器
    """
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.discovered_models = {}
    
    def discover_models(self):
        """
        自动发现所有注意力机制模型
        """
        print("正在发现模型文件...")
        
        # 查找所有Python文件
        python_files = glob.glob(os.path.join(self.model_dir, '*.py'))
        
        # 排除系统文件
        exclude_files = ['dataset.py', 'trainer.py', 'evaluator.py', 'train_all_models.py', 'test_environment.py']
        python_files = [f for f in python_files if os.path.basename(f) not in exclude_files]
        
        for file_path in python_files:
            try:
                self._load_model_from_file(file_path)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
        
        print(f"发现 {len(self.discovered_models)} 个模型")
        return self.discovered_models
    
    def _load_model_from_file(self, file_path):
        """
        从文件中加载模型类
        """
        file_name = os.path.basename(file_path)
        module_name = file_name[:-3]  # 移除.py扩展名
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"执行模块 {module_name} 时出错: {e}")
            return
        
        # 查找注意力机制类
        attention_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, nn.Module) and 
                obj != nn.Module and
                ('Attention' in name or 'attention' in name.lower()) and
                hasattr(obj, '__init__')):
                attention_classes.append((name, obj))
        
        # 存储发现的模型
        if attention_classes:
            self.discovered_models[module_name] = {
                'file_path': file_path,
                'module': module,
                'attention_classes': attention_classes
            }
            print(f"从 {file_name} 发现: {[name for name, _ in attention_classes]}")


def get_model_parameters(attention_class):
    """
    自动推断模型参数
    """
    sig = inspect.signature(attention_class.__init__)
    params = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        # 根据参数名推断默认值
        if param.default != inspect.Parameter.empty:
            params[param_name] = param.default
        else:
            # 常见参数的默认值
            if 'channel' in param_name.lower():
                params[param_name] = 128
            elif 'reduction' in param_name.lower():
                params[param_name] = 16
            elif 'kernel_size' in param_name.lower():
                params[param_name] = 7
            elif 'dim' in param_name.lower():
                params[param_name] = 128
            elif 'head' in param_name.lower():
                params[param_name] = 8
            elif 'dropout' in param_name.lower():
                params[param_name] = 0.1
            elif 'embed_dim' in param_name.lower():
                params[param_name] = 128
            elif 'num_heads' in param_name.lower():
                params[param_name] = 8
            elif 'seq_len' in param_name.lower():
                params[param_name] = 49  # 7x7 feature map
            elif 'input_dim' in param_name.lower():
                params[param_name] = 128
            elif 'hidden_dim' in param_name.lower():
                params[param_name] = 64
            else:
                # 尝试一些常见的默认值
                params[param_name] = 128  # 默认通道数
    
    return params


def create_safe_model(attention_class, model_name):
    """
    安全地创建模型实例
    """
    try:
        # 获取参数
        params = get_model_parameters(attention_class)
        
        # 尝试创建注意力模块
        attention_module = attention_class(**params)
        
        # 调用初始化权重方法（如果存在）
        if hasattr(attention_module, 'init_weights'):
            attention_module.init_weights()
        
        # 创建完整模型
        model = AttentionClassifier(
            attention_module=attention_module,
            input_channels=3,
            num_classes=10,
            feature_dim=128
        )
        
        return model, params
    
    except Exception as e:
        print(f"创建模型 {model_name} 失败: {e}")
        # 尝试使用更简单的参数
        try:
            # 只使用必需参数
            sig = inspect.signature(attention_class.__init__)
            required_params = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.default == inspect.Parameter.empty:
                    # 必需参数，使用默认值
                    if 'channel' in param_name.lower():
                        required_params[param_name] = 128
                    elif 'dim' in param_name.lower():
                        required_params[param_name] = 128
                    else:
                        required_params[param_name] = 128
            
            attention_module = attention_class(**required_params)
            
            # 调用初始化权重方法（如果存在）
            if hasattr(attention_module, 'init_weights'):
                attention_module.init_weights()
            
            model = AttentionClassifier(
                attention_module=attention_module,
                input_channels=3,
                num_classes=10,
                feature_dim=128
            )
            
            return model, required_params
        
        except Exception as e2:
            print(f"使用简化参数创建模型 {model_name} 也失败: {e2}")
            return None, None


def train_single_model(model_name, attention_class, train_loader, test_loader, 
                      epochs=10, save_dir='./models'):
    """
    训练单个模型
    """
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model, params = create_safe_model(attention_class, model_name)
    if model is None:
        return None
    
    print(f"模型参数: {params}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_name}.pth")
    
    try:
        # 训练模型
        trainer = ModelTrainer(model)
        train_results = trainer.train(
            train_loader, test_loader, 
            epochs=epochs, 
            save_path=model_save_path
        )
        
        # 评估模型
        evaluator = ModelEvaluator(model)
        eval_results = evaluator.evaluate(test_loader)
        
        # 合并结果
        results = {
            **train_results,
            **eval_results,
            'model_name': model_name,
            'model_parameters': params,
            'model_path': model_save_path
        }
        
        # 保存结果
        results_path = os.path.join(save_dir, f"{model_name}_results.json")
        evaluator.save_results(eval_results, results_path)
        
        print(f"模型 {model_name} 训练完成!")
        print(f"最终准确率: {eval_results['accuracy']:.2f}%")
        
        return results
    
    except Exception as e:
        print(f"训练模型 {model_name} 时出错: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='训练所有注意力机制模型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--max_models', type=int, default=None, help='最大训练模型数量')
    parser.add_argument('--model_filter', type=str, default=None, help='模型名称过滤器')
    
    args = parser.parse_args()
    
    print("MNIST注意力机制模型训练系统")
    print("="*60)
    
    # 获取数据
    print("加载MNIST数据集...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=args.batch_size)
    print(f"训练集: {len(train_loader)} 批次")
    print(f"测试集: {len(test_loader)} 批次")
    
    # 发现模型
    discovery = ModelDiscovery()
    models = discovery.discover_models()
    
    if not models:
        print("未发现任何模型!")
        return
    
    # 过滤模型
    if args.model_filter:
        models = {k: v for k, v in models.items() if args.model_filter.lower() in k.lower()}
        print(f"应用过滤器 '{args.model_filter}', 剩余 {len(models)} 个模型")
    
    # 限制模型数量
    if args.max_models:
        models = dict(list(models.items())[:args.max_models])
        print(f"限制训练模型数量为 {args.max_models}")
    
    # 训练所有模型
    all_results = {}
    successful_models = 0
    failed_models = 0
    
    start_time = datetime.now()
    
    for model_name, model_info in models.items():
        for class_name, attention_class in model_info['attention_classes']:
            full_model_name = f"{model_name}_{class_name}"
            
            result = train_single_model(
                full_model_name, attention_class, 
                train_loader, test_loader,
                epochs=args.epochs,
                save_dir=args.save_dir
            )
            
            if result:
                all_results[full_model_name] = result
                successful_models += 1
            else:
                failed_models += 1
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # 生成总结报告
    print("\n" + "="*80)
    print("训练总结报告")
    print("="*80)
    print(f"总训练时间: {total_time}")
    print(f"成功训练模型: {successful_models}")
    print(f"失败模型: {failed_models}")
    print(f"总模型数: {successful_models + failed_models}")
    
    if all_results:
        # 比较模型性能
        compare_models(all_results)
        
        # 保存总结报告
        summary_path = os.path.join(args.save_dir, 'training_summary.json')
        summary = {
            'timestamp': start_time.isoformat(),
            'total_time': str(total_time),
            'successful_models': successful_models,
            'failed_models': failed_models,
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            },
            'results': {k: {
                'accuracy': v.get('accuracy', 0),
                'loss': v.get('loss', 0),
                'total_time': v.get('total_time', 0)
            } for k, v in all_results.items()}
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练总结已保存到: {summary_path}")
    
    print("\n所有模型训练完成!")


if __name__ == "__main__":
    main()
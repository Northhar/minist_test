import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
from typing import Dict, List, Tuple
import json


class ModelEvaluator:
    """
    模型评估器
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self, test_loader, class_names=None):
        """
        全面评估模型性能
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 获取预测结果
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                correct += predictions.eq(target).sum().item()
                total += target.size(0)
                
                if batch_idx % 50 == 0:
                    print(f'评估进度: {batch_idx}/{len(test_loader)} batches')
        
        evaluation_time = time.time() - start_time
        
        # 计算指标
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # 生成详细报告
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total,
            'correct_predictions': correct,
            'evaluation_time': evaluation_time,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # 分类报告
        if class_names is None:
            class_names = [str(i) for i in range(10)]  # MNIST有10个类别
        
        classification_rep = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        results['classification_report'] = classification_rep
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        results['confusion_matrix'] = conf_matrix
        
        return results
    
    def print_evaluation_results(self, results):
        """
        打印评估结果
        """
        print("=" * 60)
        print("模型评估结果")
        print("=" * 60)
        print(f"总体准确率: {results['accuracy']:.2f}%")
        print(f"平均损失: {results['loss']:.4f}")
        print(f"总样本数: {results['total_samples']}")
        print(f"正确预测数: {results['correct_predictions']}")
        print(f"评估时间: {results['evaluation_time']:.2f}秒")
        print()
        
        # 打印分类报告
        print("分类报告:")
        print("-" * 40)
        classification_rep = results['classification_report']
        
        # 打印每个类别的指标
        for class_name in [str(i) for i in range(10)]:
            if class_name in classification_rep:
                metrics = classification_rep[class_name]
                print(f"类别 {class_name}: "
                      f"精确率={metrics['precision']:.3f}, "
                      f"召回率={metrics['recall']:.3f}, "
                      f"F1分数={metrics['f1-score']:.3f}")
        
        # 打印总体指标
        if 'macro avg' in classification_rep:
            macro_avg = classification_rep['macro avg']
            print(f"\n宏平均: "
                  f"精确率={macro_avg['precision']:.3f}, "
                  f"召回率={macro_avg['recall']:.3f}, "
                  f"F1分数={macro_avg['f1-score']:.3f}")
        
        if 'weighted avg' in classification_rep:
            weighted_avg = classification_rep['weighted avg']
            print(f"加权平均: "
                  f"精确率={weighted_avg['precision']:.3f}, "
                  f"召回率={weighted_avg['recall']:.3f}, "
                  f"F1分数={weighted_avg['f1-score']:.3f}")
    
    def plot_confusion_matrix(self, results, save_path=None, class_names=None):
        """
        绘制混淆矩阵
        """
        if class_names is None:
            class_names = [str(i) for i in range(10)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def analyze_errors(self, results, test_loader, num_examples=10):
        """
        分析错误预测的样本
        """
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        probabilities = np.array(results['probabilities'])
        
        # 找到错误预测的索引
        error_indices = np.where(predictions != targets)[0]
        
        if len(error_indices) == 0:
            print("没有错误预测！")
            return
        
        print(f"总错误数: {len(error_indices)}")
        print(f"错误率: {len(error_indices)/len(targets)*100:.2f}%")
        
        # 分析每个类别的错误
        print("\n各类别错误分析:")
        for class_id in range(10):
            class_errors = error_indices[targets[error_indices] == class_id]
            if len(class_errors) > 0:
                class_total = np.sum(targets == class_id)
                error_rate = len(class_errors) / class_total * 100
                print(f"类别 {class_id}: {len(class_errors)}/{class_total} 错误 ({error_rate:.1f}%)")
        
        # 显示一些错误样本（如果有图像数据的话）
        print(f"\n显示前 {min(num_examples, len(error_indices))} 个错误样本:")
        for i in range(min(num_examples, len(error_indices))):
            idx = error_indices[i]
            true_label = targets[idx]
            pred_label = predictions[idx]
            confidence = probabilities[idx][pred_label]
            print(f"样本 {idx}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f}")
    
    def save_results(self, results, save_path):
        """
        保存评估结果到文件
        """
        # 准备可序列化的结果
        serializable_results = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'total_samples': results['total_samples'],
            'correct_predictions': results['correct_predictions'],
            'evaluation_time': results['evaluation_time'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {save_path}")


def compare_models(model_results: Dict[str, Dict]):
    """
    比较多个模型的性能
    """
    print("=" * 80)
    print("模型性能比较")
    print("=" * 80)
    
    # 创建比较表格
    print(f"{'模型名称':<20} {'准确率':<10} {'损失':<10} {'F1分数':<10} {'训练时间':<10}")
    print("-" * 70)
    
    for model_name, results in model_results.items():
        accuracy = results.get('accuracy', 0)
        loss = results.get('loss', 0)
        
        # 获取宏平均F1分数
        f1_score = 0
        if 'classification_report' in results:
            macro_avg = results['classification_report'].get('macro avg', {})
            f1_score = macro_avg.get('f1-score', 0)
        
        eval_time = results.get('evaluation_time', 0)
        
        print(f"{model_name:<20} {accuracy:<10.2f} {loss:<10.4f} {f1_score:<10.3f} {eval_time:<10.2f}")
    
    # 找出最佳模型
    best_model = max(model_results.items(), key=lambda x: x[1].get('accuracy', 0))
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.2f}%)")


if __name__ == "__main__":
    # 测试评估器
    # 修正导入路径：从同目录 trainer 导入 create_model_with_attention，并从同目录 dataset 导入数据加载器
    from trainer import create_model_with_attention
    from dataset import get_mnist_dataloaders
    
    # 创建一个简单的测试模型
    class SimpleAttention(nn.Module):
        def __init__(self, channel=128):
            super().__init__()
            self.channel = channel
        
        def forward(self, x):
            return x
    
    # 获取数据
    _, test_loader = get_mnist_dataloaders(batch_size=64)
    
    # 创建模型
    model = create_model_with_attention(SimpleAttention, {'channel': 128})
    
    # 创建评估器
    evaluator = ModelEvaluator(model)
    
    # 评估模型
    results = evaluator.evaluate(test_loader)
    evaluator.print_evaluation_results(results)
    evaluator.analyze_errors(results, test_loader)
# SEAttention 模型训练

这个文件夹包含了用于训练 SENet (Squeeze-and-Excitation Networks) 注意力机制的独立训练脚本。

## 文件结构

- `model.py`: 包含 SEAttention 模型的实现
- `trainer.py`: 训练脚本，包含完整的训练流程
- `README.md`: 使用说明

## 模型说明

SEAttention 是一种通道注意力机制，通过以下步骤工作：

1. **Squeeze**: 使用全局平均池化将空间维度压缩为 1×1
2. **Excitation**: 使用两个全连接层学习通道间的依赖关系
3. **Scale**: 将学习到的权重应用到原始特征图上

## 使用方法

### 直接运行训练

```bash
cd 1_1_SENet
python trainer.py
```

### 自定义参数

可以修改 `trainer.py` 中的以下参数：

- `channel`: SEAttention 的通道数 (默认: 128)
- `reduction`: 降维比例 (默认: 16)
- `epochs`: 训练轮数 (默认: 5)
- `batch_size`: 批次大小 (默认: 64)

## 训练输出

训练过程会显示：
- 每个epoch的训练和验证损失
- 每个epoch的训练和验证准确率
- 最佳模型会保存为 `best_senet_model.pth`

## 依赖项

- PyTorch
- 父目录中的 `dataset.py` 和 `evaluator.py`
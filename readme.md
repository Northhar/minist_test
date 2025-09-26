项目说明（README）

环境要求
- Python 版本：建议 3.10（项目中出现的 pycache 为 cpython-310）
- 深度学习框架：PyTorch（GPU 可选）
- CUDA：可选，用于加速（当前环境未使用cuda）

目录结构与内容
- models/
  - 汇总各类模型脚本的目录，采用“编号_名称.py”的命名方式，例如：1_1_SENet.py、2_10_MCANet.py 等。
  - 提示：权重文件不建议存放在此目录（当前已清理），如需保存本地权重请使用 *.pth 并在 .gitignore 中忽略。

- 1_1_SENet/
  - 一个可独立运行的 SENet 示例工程，包含：
    - model.py：SENet 与 SE 注意力模块的网络结构实现
    - trainer.py：训练流程（数据加载、优化器、损失函数、训练循环等）
    - dataset.py：MNIST 数据集读取与预处理逻辑（含通道处理）
    - evaluator.py：评估与指标计算（可视化或测试）
    - best_senet_model.pth：已训练的最佳模型权重
    - README.md：该子模块的使用说明

- Mamba_file/
  - 与 Mamba/SSM 相关的模型与工具集合，包含：
    - csrc/selective_scan/：选择性扫描算子相关源码
    - group_mamba/：组 Mamba 实现（如 csms6s.py、ss2d.py）
    - mamba_ssm_self/：包含 models、modules、ops、utils 等子目录
    - VMRNN_self/vmamba.py：VMamba 参考实现

- archive/
  - MNIST 数据集的原始 IDX 文件（如未通过脚本自动下载，可手动放置在该目录）：
    - train-images.idx3-ubyte、train-labels.idx1-ubyte
    - t10k-images.idx3-ubyte、t10k-labels.idx1-ubyte

- MNIST_ran_models.md
  - 记录已经在 MNIST 数据集上跑过的模型列表（当前：1_1_SENet）。

- 项目根目录下的通用脚本
  - dataset.py：通用的 MNIST 数据加载与预处理工具（与独立子模块中的 dataset.py 相区分，主要用于根目录训练流程）
  - trainer.py：通用训练脚本，可与 models 目录下的模型脚本联动（指定模型后进行训练）
  - evaluator.py：通用评估脚本，用于计算指标、测试集评估等
  - train_all_models.py：批量训练/测试多个模型的入口脚本
  - test_environment.py：环境与依赖检测脚本（包含模块导入测试等）

- 其它文件与目录
  - .gitignore：忽略不需要纳入版本控制的内容（当前已包含 __pycache__/ 与 .idea/）
  - .idea/：IDE 项目配置目录（已在 .gitignore 中忽略）
  - __pycache__/：Python 字节码缓存（已在 .gitignore 中忽略）

快速开始
- 准备环境：安装 Python 3.10 与 PyTorch（GPU 可选，CUDA 11.1 可用于加速）
- 训练独立 SENet 示例：
  - 进入 1_1_SENet 目录，运行 trainer.py
- 使用通用训练脚本：
  - 在项目根目录运行 trainer.py，并指定目标模型脚本（例如 models/1_1_SENet.py）

注意事项
- 数据通道数：MNIST 为单通道，如果数据预处理将其堆叠为 3 通道，请确保模型的输入通道设置与数据保持一致，否则会出现维度不匹配错误。
- 权重文件管理：建议将本地权重文件（*.pth）置于专用目录并在 .gitignore 中忽略，避免仓库体积过大。
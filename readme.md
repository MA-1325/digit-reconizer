#  CNN 手写数字识别系统（MNIST）

本项目基于 PyTorch 实现卷积神经网络（CNN）对 MNIST 手写数字数据集进行分类识别，旨在探索轻量化模型设计、训练优化策略与模型版本管理实践。
##  核心技术  
- PyTorch
- CNN

##  项目亮点

- 使用 PyTorch 自定义构建 CNN 模型，结构简洁高效
- 在 MNIST 数据集上实现 **99.2%** 的验证准确率
- 应用数据增强、Dropout、学习率调节等策略提升模型泛化能力
- 利用 Git 管理模型版本，支持多轮实验对比与复现
- 成功在 Kaggle 平台提交多个版本并获得精度提升

##  项目结构
- digit.py	主要训练脚本。 包含 CNN 模型定义、数据加载、训练循环、评估和最终生成提交文件的逻辑
- Figure_1.png	项目的可视化结果，可能是损失/准确率曲线或模型结构图
- test.csv / train.csv	原始数据集文件。 训练集和测试集数据，通常包含像素值和标签（train.csv）
- sample_submission.csv	Kaggle 提供的提交示例文件
- submission_pytorch.csv	最终预测结果文件。 由 digit.py 生成，用于向 Kaggle 提交结果
- requirements.txt	项目所需的 Python 依赖库及其版本

## 3. 环境配置与安装

### 3.1 依赖库安装

本项目基于 Python 3.x 和 PyTorch。请确保您已安装所有必需的依赖库：

```bash
pip install -r requirements.txt
```
##  4. 运行与训练 

###  4.1 训练模型

运行主脚本开始模型训练。脚本将加载数据、训练 CNN 模型并在训练结束后生成预测结果。

```bash
python digit.py
```
##  您可以在 digit.py 中修改以下关键参数：

EPOCHS: 训练轮数（例如 20）

BATCH_SIZE: 批次大小（例如 64）

LEARNING_RATE: 学习率（例如 0.01）

---

## 5. 模型性能与结果

### 5.1 验证集准确率

- **最佳准确率**: 99.2%
- **Kaggle Public Score**: 0.99242

### 5.2 训练可视化
`Figure_1.png` 展示了模型的训练损失和验证准确率曲线。

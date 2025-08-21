```md
# Spectral Attention UNet++ 项目

本项目基于 **Spectral Attention UNet++ 网络**，用于训练和推理。  

## 📂 项目结构
```

Project/
├── Code/                # 代码目录
│   ├── train.py         # 训练脚本
│   ├── predict.py       # 推理脚本
│   ├── requirement.txt  # Python 依赖
│   └── ...              # 其他源码
├── Data/                # 数据目录
│   ├── Train/           # 训练数据
│   └── Infer/           # 推理数据
└── README.md

````

---

## ⚙️ 环境配置

建议使用 anaconda/miniconda 创建虚拟环境：

```bash
# 创建环境（Python 3.8）
conda create -n UNet python=3.8

# 激活环境
conda activate UNet

# 安装依赖
pip install -r requirement.txt
````

---

## 🚀 使用方法

### 1. 训练模型

```bash
cd Code
python train.py
```

### 2. 模型推理

```bash
cd Code
python predict.py
```

---

## 📊 数据说明

* `Data/Train/`
  存放训练所需的数据集。

* `Data/Infer/`
  存放需要进行推理/预测的数据。

---

## 📝 说明

* 请根据需要修改 `train.py` 和 `predict.py` 中的参数（如路径、超参数）。
* 输出结果将保存在脚本指定的目录中。

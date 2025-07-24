# PyTorchCPP 项目结构

本文档详细说明了 PyTorchCPP 项目的目录结构和各个组件的功能。

## 顶层目录结构

```
pytorch-cpp/
├── include/           # 公共头文件目录
├── src/               # 源代码实现
├── bindings/          # Python绑定代码
├── tests/             # 测试代码
├── examples/          # 示例代码
├── docs/              # 文档
├── build/             # 构建目录（由CMake生成）
├── CMakeLists.txt     # 根CMake配置文件
├── setup.py           # Python安装脚本
├── README.md          # 项目简介
└── .gitignore         # Git忽略文件
```

## include/ 目录

包含所有公共头文件，按照模块分组，使用与命名空间相符的目录结构：

```
include/
└── pytorchcpp/       # 主命名空间目录
    ├── tensor.h      # 张量模块
    ├── autograd.h    # 自动求导模块
    ├── nn.h          # 神经网络模块
    └── optim.h       # 优化器模块
```

## src/ 目录

包含各个模块的具体实现，与头文件的目录结构对应：

```
src/
├── tensor/            # 张量模块实现
│   ├── CMakeLists.txt
│   └── tensor.cpp
├── autograd/          # 自动求导模块实现
│   ├── CMakeLists.txt
│   ├── function.cpp
│   └── variable.cpp
├── nn/                # 神经网络模块实现
│   ├── CMakeLists.txt
│   ├── module.cpp
│   ├── linear.cpp
│   ├── activation.cpp
│   ├── sequential.cpp
│   └── loss.cpp
├── optim/             # 优化器模块实现
│   ├── CMakeLists.txt
│   ├── optimizer.cpp
│   ├── sgd.cpp
│   └── adam.cpp
└── CMakeLists.txt     # 源码主CMake文件
```

## bindings/ 目录

包含Python绑定代码，使用pybind11库：

```
bindings/
├── CMakeLists.txt
└── pybind/
    ├── CMakeLists.txt
    ├── bindings.cpp        # 主绑定模块
    ├── tensor_bindings.cpp # 张量模块绑定
    ├── autograd_bindings.cpp # 自动求导模块绑定
    ├── nn_bindings.cpp     # 神经网络模块绑定
    └── optim_bindings.cpp  # 优化器模块绑定
```

## tests/ 目录

包含测试代码，使用Google Test框架：

```
tests/
├── CMakeLists.txt
└── cpp/                    # C++测试
    ├── CMakeLists.txt
    ├── tensor_test.cpp
    ├── autograd_test.cpp
    ├── nn_test.cpp
    └── optim_test.cpp
```

## examples/ 目录

包含示例代码，展示API的用法：

```
examples/
├── CMakeLists.txt
├── cpp/               # C++示例
│   ├── CMakeLists.txt
│   ├── matrix_ops.cpp         # 矩阵操作示例
│   ├── autograd_example.cpp   # 自动求导示例
│   ├── linear_regression.cpp  # 线性回归示例
│   └── mnist_classification.cpp # MNIST分类示例
└── python/           # Python示例
    └── linear_regression.py   # Python线性回归示例
```

## docs/ 目录

包含项目文档：

```
docs/
├── build_instructions.md     # 编译和运行说明
├── common_errors.md          # 常见错误及解决方案
├── cmake_explanation.md      # CMake系统详解
└── project_structure.md      # 本文档
```

## 核心组件概述

### 1. 张量模块 (Tensor)

张量模块是框架的基础，提供多维数组的表示和操作。

**关键特性：**
- 多维张量的创建和操作
- 基本的线性代数操作（矩阵乘法、转置等）
- 元素级操作（加、减、乘、除等）
- 形状操作（reshape、view等）
- 支持梯度计算标记（requires_grad）

**实现细节：**
- 使用PIMPL（指向实现的指针）模式隐藏实现细节
- 基于`std::vector<float>`的内存布局
- 支持索引计算和元素访问

### 2. 自动求导模块 (Autograd)

自动求导模块实现了自动微分，是神经网络反向传播的基础。

**关键特性：**
- Variable类：封装Tensor并跟踪计算历史
- Function类：表示计算图中的操作节点
- 支持前向和反向传播
- 自动梯度累积

**实现细节：**
- 动态构建计算图
- 每个操作都有对应的Function子类实现前向和反向计算
- 支持基本操作（加法、乘法、矩阵乘法等）的梯度计算

### 3. 神经网络模块 (NN)

神经网络模块提供构建神经网络模型的组件。

**关键特性：**
- Module基类：所有神经网络层的基础
- 常见层实现：Linear（全连接层）
- 激活函数：ReLU、Sigmoid、Tanh
- Sequential容器：按顺序堆叠多个层
- 损失函数：MSELoss、CrossEntropyLoss

**实现细节：**
- 模块化设计，允许轻松组合不同的层
- 每个模块包含参数和前向传播逻辑
- 支持参数管理和梯度传递

### 4. 优化器模块 (Optim)

优化器模块实现神经网络参数的更新策略。

**关键特性：**
- Optimizer基类：所有优化器的基础
- SGD（随机梯度下降）：支持动量和权重衰减
- Adam：自适应矩估计优化器

**实现细节：**
- 参数引用管理
- 支持梯度清零和参数更新
- 针对不同优化算法的特定状态管理

### 5. Python绑定 (pybind11)

使用pybind11提供对C++库的Python接口。

**关键特性：**
- 暴露所有核心C++类和函数
- 与Python习惯对齐的接口
- 类型转换（如Tensor与NumPy互操作）

**实现细节：**
- 模块化绑定代码
- 适当的类型转换和内存管理
- 异常处理和错误传播

## 依赖关系

组件间的依赖关系如下：

1. **Tensor**：核心组件，不依赖其他模块
2. **Autograd**：依赖Tensor
3. **NN**：依赖Tensor和Autograd
4. **Optim**：依赖Tensor和Autograd
5. **Python绑定**：依赖所有其他模块

这种分层设计使各模块职责清晰，便于维护和扩展。

## 扩展指南

向项目添加新功能时，应遵循以下步骤：

1. **确定适当的模块**：新功能应该添加到哪个模块？
2. **在头文件中声明API**：在适当的头文件中添加公共接口
3. **实现功能**：在对应模块的源文件中添加实现
4. **添加测试**：在tests/cpp/目录中添加测试用例
5. **更新CMake**：确保新文件被包含在构建系统中
6. **添加Python绑定**：在bindings/pybind/目录中添加相应的绑定代码
7. **创建示例**：在examples/目录中添加示例代码，展示新功能的用法
8. **更新文档**：记录新功能的用法和实现细节

通过遵循项目的结构和命名约定，新功能可以无缝集成到现有框架中。 
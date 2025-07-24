# PyTorchCPP 编译与运行说明

本文档提供了如何编译和运行 PyTorchCPP 项目的详细说明。

## 前置条件

确保你的系统已经安装以下组件：

- C++17兼容的编译器 (如 GCC 7.3+, Clang 5.0+)
- CMake 3.14+
- Python 3.7+ (用于Python绑定)
- Git

## 获取代码

```bash
# 克隆仓库
git clone https://github.com/shizhengLi/pytorch-cpp.git
cd pytorch-cpp
```

## C++ 库编译

### 方法 1: 命令行构建

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置CMake
cmake ..

# 编译
make -j

# 或者指定使用多少个核心进行编译
# make -j4
```

### 方法 2: 使用CMake预设

```bash
# 查看可用的预设
cmake --list-presets

# 使用Debug预设
cmake --preset=debug
cmake --build --preset=debug

# 使用Release预设
cmake --preset=release
cmake --build --preset=release
```

## 运行示例

编译完成后，可以运行示例程序：

```bash
# 从构建目录
cd build

# 运行矩阵运算示例
./bin/matrix_ops

# 运行自动求导示例
./bin/autograd_example

# 运行线性回归示例
./bin/linear_regression

# 运行MNIST分类示例
./bin/mnist_classification
```

## 运行测试

```bash
# 从构建目录
cd build

# 运行所有测试
ctest

# 或者运行特定的测试
./bin/tensor_test
./bin/autograd_test
./bin/nn_test
./bin/optim_test
```

## Python绑定

### 方法 1: 使用 CMake

```bash
# 从构建目录
cd build

# 构建Python模块
make pytorchcpp_python
```

### 方法 2: 使用 pip

```bash
# 从项目根目录
pip install -e .
```

### 运行Python示例

```bash
# 确保Python能找到模块
export PYTHONPATH=$PYTHONPATH:/path/to/pytorch-cpp/build/lib

# 运行Python线性回归示例
python examples/python/linear_regression.py
```

## 安装

如果希望系统级安装该库：

```bash
# 从构建目录
cd build

# 安装
make install

# 可能需要管理员权限
# sudo make install
```

## 调试构建

如果需要进行调试：

```bash
# 创建调试构建
mkdir -p build_debug && cd build_debug

# 配置CMake（调试模式）
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 编译
make -j
```

## 其他有用的CMake选项

- `-DCMAKE_INSTALL_PREFIX=/path/to/install` - 设置安装路径
- `-DBUILD_SHARED_LIBS=ON` - 构建共享库而非静态库
- `-DBUILD_TESTING=OFF` - 禁用测试构建
- `-DBUILD_PYTHON_BINDINGS=OFF` - 禁用Python绑定构建

希望这些说明能帮助你成功编译和运行 PyTorchCPP 项目！ 
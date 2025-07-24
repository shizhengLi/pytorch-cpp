# PyTorchCPP

PyTorchCPP是一个轻量级深度学习框架，使用C++实现PyTorch核心功能，并提供Python绑定。

## 项目特点

- 高性能的张量计算
- 自动微分系统
- 神经网络模块
- 优化器支持
- Python绑定

## 目标

这个项目旨在帮助理解深度学习框架的底层实现，特别是PyTorch的C++后端和Python绑定机制。

## 安装

### 前置条件

- C++17兼容的编译器
- CMake 3.14+
- Python 3.7+（用于Python绑定）

### 构建和安装

```bash
# 克隆仓库
git clone https://github.com/shizhengLi/pytorch-cpp.git
cd pytorch-cpp

# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake ..

# 编译
make -j

# 安装（可选）
make install
```

### Python绑定

```bash
# 在build目录中构建Python模块
make pytorchcpp_python

# 或者使用pip安装
pip install -e .
```

## 快速入门

### C++示例

```cpp
#include <pytorchcpp/tensor.h>
#include <pytorchcpp/nn/linear.h>
#include <pytorchcpp/optim/sgd.h>

using namespace pytorchcpp;

int main() {
    // 创建张量
    auto x = Tensor::ones({2, 3});
    auto y = Tensor::randn({3, 2});
    
    // 矩阵乘法
    auto z = x.matmul(y);
    std::cout << "z = " << z << std::endl;
    
    // 创建线性层
    auto linear = nn::Linear(10, 5);
    
    // 前向传播
    auto input = Tensor::randn({1, 10});
    auto output = linear->forward(input);
    
    return 0;
}
```

### Python示例

```python
import pytorchcpp as ptc

# 创建张量
x = ptc.Tensor.ones([2, 3])
y = ptc.Tensor.randn([3, 2])

# 矩阵乘法
z = x.matmul(y)
print(f"z = {z}")

# 创建线性层
linear = ptc.nn.Linear(10, 5)

# 前向传播
input = ptc.Tensor.randn([1, 10])
output = linear(input)
```

## 文档

详细文档可以在[docs](docs/)目录中找到。

## 测试

```bash
# 在build目录中运行测试
make test
```

## 贡献

欢迎贡献代码和反馈问题。请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解更多信息。

## 许可证

本项目采用MIT许可证。详细信息请参见[LICENSE](LICENSE)文件。 
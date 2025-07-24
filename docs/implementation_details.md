# PyTorchCPP 实现细节

本文档详细介绍 PyTorchCPP 项目的核心算法和实现方式，帮助开发者理解框架内部的工作原理。

## 目录

1. [张量实现](#张量实现)
2. [自动求导系统](#自动求导系统)
3. [神经网络模块](#神经网络模块)
4. [优化器实现](#优化器实现)
5. [Python 绑定](#python-绑定)

## 张量实现

### PIMPL 设计模式

PyTorchCPP 的 `Tensor` 类使用 PIMPL（指向实现的指针，Pointer to IMPLementation）设计模式，这种模式将实现细节隐藏在私有的 `Impl` 结构体中：

```cpp
// include/pytorchcpp/tensor.h
class Tensor {
public:
    // 公共接口...
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

// src/tensor/tensor.cpp
struct Tensor::Impl {
    std::vector<size_t> shape;
    std::vector<float> data;
    bool requires_grad;
    
    // 实现细节...
};
```

这种设计的优点：
- 降低编译依赖性：修改实现细节不需要重新编译使用 `Tensor` 的代码
- 信息隐藏：使用者只能通过公共接口访问 `Tensor`
- 实现可以随时改变而不影响 API

### 内存布局

`Tensor` 使用一维 `std::vector<float>` 来存储多维数据，并通过索引计算来访问元素：

```cpp
size_t Tensor::Impl::index_to_offset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Indices dimension mismatch");
    }
    
    size_t offset = 0;
    size_t multiplier = 1;
    
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        offset += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    
    return offset;
}
```

这里使用了行优先存储（row-major）顺序，与 C/C++ 数组的存储方式一致。

### 工厂方法

`Tensor` 类提供了几种工厂方法来创建特殊张量：

```cpp
// 全零张量
Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);  // 默认初始化为0
}

// 全一张量
Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 
                                 static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> data(size, 1.0f);
    return Tensor(shape, data, requires_grad);
}

// 随机正态分布张量
Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 
                                 static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> data(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : data) {
        val = dist(gen);
    }
    
    return Tensor(shape, data, requires_grad);
}
```

### 数学运算

张量的数学运算按照元素类型分为两类：

1. **元素级操作**：如加法、乘法等，对应元素直接运算

```cpp
Tensor Tensor::add(const Tensor& other) const {
    if (pImpl->shape != other.pImpl->shape) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    std::vector<float> result_data(pImpl->data.size());
    for (size_t i = 0; i < pImpl->data.size(); ++i) {
        result_data[i] = pImpl->data[i] + other.pImpl->data[i];
    }
    
    return Tensor(pImpl->shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}
```

2. **矩阵运算**：如矩阵乘法、转置等，需要特殊的算法

```cpp
Tensor Tensor::matmul(const Tensor& other) const {
    // 简单实现2D矩阵乘法
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    size_t m = pImpl->shape[0];
    size_t k = pImpl->shape[1];
    
    if (k != other.pImpl->shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    
    size_t n = other.pImpl->shape[1];
    std::vector<size_t> result_shape = {m, n};
    std::vector<float> result_data(m * n, 0.0f);
    
    // 实现简单的矩阵乘法
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                result_data[i * n + j] += pImpl->data[i * k + k_idx] * 
                                          other.pImpl->data[k_idx * n + j];
            }
        }
    }
    
    return Tensor(result_shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}
```

注意：这是简单实现，实际的高性能张量库可能会使用BLAS（基本线性代数子程序）等优化库。

## 自动求导系统

### 计算图结构

PyTorchCPP 实现了动态计算图：

1. **Variable 类**：封装 `Tensor` 并跟踪梯度信息
2. **Function 类**：表示计算图中的操作节点

```cpp
// Variable的关键部分
struct Variable::Impl {
    Tensor data;              // 存储数据
    Tensor grad;              // 存储梯度
    bool requires_grad;       // 是否需要梯度
    std::shared_ptr<Function> grad_fn;  // 创建该变量的操作
    // ...
};

// Function基类
class Function {
public:
    virtual Variable forward(const std::vector<Variable>& inputs) = 0;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
protected:
    std::vector<Variable> inputs;  // 输入变量
    Variable output;              // 输出变量
    
    // 保存前向传播的输入和输出，用于反向传播
    void save_for_backward(const std::vector<Variable>& inputs, const Variable& output) {
        this->inputs = inputs;
        this->output = output;
    }
};
```

### 自动求导示例

以加法为例，展示自动求导的实现：

```cpp
// 加法函数的定义
class AddFunction : public Function {
public:
    Variable forward(const std::vector<Variable>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("AddFunction requires 2 inputs");
        }
        
        // 计算输出
        Tensor result = inputs[0].data() + inputs[1].data();
        Variable output(result, inputs[0].requires_grad() || inputs[1].requires_grad());
        
        // 保存输入和输出用于反向传播
        save_for_backward(inputs, output);
        
        return output;
    }
    
    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // 对于加法，梯度直接传递给两个输入
        std::vector<Tensor> grad_inputs(2);
        
        if (inputs[0].requires_grad()) {
            // 第一个输入的梯度等于输出的梯度
            grad_inputs[0] = grad_output;
        }
        
        if (inputs[1].requires_grad()) {
            // 第二个输入的梯度也等于输出的梯度
            grad_inputs[1] = grad_output;
        }
        
        return grad_inputs;
    }
};
```

变量操作会创建相应的函数并设置梯度函数：

```cpp
Variable Variable::add(const Variable& other) const {
    auto add_fn = std::make_shared<AddFunction>();
    auto result = add_fn->forward({*this, other});
    result.pImpl->grad_fn = add_fn;  // 设置梯度函数
    return result;
}
```

### 反向传播实现

`backward()` 方法是自动求导的核心，它从输出变量开始，递归地将梯度传播到所有需要梯度的输入变量：

```cpp
void Variable::backward(const Tensor& grad) {
    if (!pImpl->requires_grad) {
        throw std::runtime_error("Cannot call backward on a variable that doesn't require gradient");
    }
    
    // 如果未提供梯度，则使用全1张量
    Tensor gradient = grad;
    if (grad.numel() == 0) {
        if (pImpl->data.numel() == 1) {
            gradient = Tensor::ones({1}, false);
        } else {
            gradient = Tensor::ones(pImpl->data.shape(), false);
        }
    }
    
    // 累积梯度
    pImpl->grad = pImpl->grad + gradient;
    
    // 如果有梯度函数，则继续反向传播
    if (pImpl->grad_fn) {
        // 计算输入梯度
        auto grad_inputs = pImpl->grad_fn->backward(gradient);
        
        // 继续传播到输入变量
        size_t i = 0;
        for (const auto& input : pImpl->grad_fn->inputs) {
            if (input.requires_grad() && i < grad_inputs.size()) {
                // 递归调用输入变量的backward，传递其梯度
                const_cast<Variable&>(input).backward(grad_inputs[i]);
            }
            ++i;
        }
    }
}
```

## 神经网络模块

### 模块层次结构

神经网络模块基于 `Module` 基类构建层次结构：

```cpp
class Module {
public:
    virtual ~Module() = default;
    
    // 前向传播
    virtual Variable forward(const Variable& input) = 0;
    
    // 简化调用
    Variable operator()(const Variable& input) {
        return forward(input);
    }
    
    // 收集所有参数
    virtual std::unordered_map<std::string, Variable> parameters() const {
        return params_;
    }
    
protected:
    std::unordered_map<std::string, Variable> params_;
    bool training_ = true;
    
    // 参数注册
    void register_parameter(const std::string& name, const Variable& param) {
        params_[name] = param;
    }
};
```

### 全连接层实现

`Linear` 类实现了全连接层（又称线性层或密集层）：

```cpp
Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
    
    // 权重初始化 (He初始化)
    float stdv = std::sqrt(2.0f / in_features);
    
    // 创建权重参数
    std::vector<float> weight_data(out_features * in_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stdv);
    
    for (auto& val : weight_data) {
        val = dist(gen);
    }
    
    Tensor weight_tensor({out_features, in_features}, weight_data);
    weight_ = Variable(weight_tensor, true);
    register_parameter("weight", weight_);
    
    // 如果需要偏置，创建偏置参数
    if (has_bias_) {
        std::vector<float> bias_data(out_features, 0.0f);
        Tensor bias_tensor({out_features}, bias_data);
        bias_ = Variable(bias_tensor, true);
        register_parameter("bias", bias_);
    }
}

Variable Linear::forward(const Variable& input) {
    auto input_shape = input.data().shape();
    if (input_shape.back() != in_features_) {
        throw std::invalid_argument("Expected input features: " + std::to_string(in_features_) +
                                   ", got: " + std::to_string(input_shape.back()));
    }
    
    // y = x * W^T + b
    auto output = input.matmul(weight_.data().transpose());
    
    if (has_bias_) {
        output = output + bias_;
    }
    
    return output;
}
```

### 激活函数

激活函数实现了非线性变换：

```cpp
Variable ReLU::forward(const Variable& input) {
    auto input_data = input.data();
    auto shape = input_data.shape();
    size_t size = input_data.numel();
    std::vector<float> result_data(size);
    
    // ReLU: max(0, x)
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(0.0f, input_data[i]);
    }
    
    return Variable(Tensor(shape, result_data), input.requires_grad());
}
```

### 损失函数

损失函数计算模型输出和目标之间的差异：

```cpp
Variable MSELoss::forward(const Variable& input, const Variable& target) {
    if (input.data().shape() != target.data().shape()) {
        throw std::invalid_argument("Input and target shapes must match for MSE loss");
    }
    
    // 计算平方差
    auto diff = input - target;
    auto squared_diff = diff * diff;
    
    // 计算平均值
    auto loss = squared_diff.data().mean();
    return Variable(loss, input.requires_grad() || target.requires_grad());
}
```

## 优化器实现

### 优化器基类

所有优化器继承自 `Optimizer` 基类：

```cpp
class Optimizer {
public:
    Optimizer(const std::unordered_map<std::string, Variable>& parameters, float learning_rate = 0.01f)
        : learning_rate_(learning_rate) {
        // 收集需要优化的参数
        for (const auto& [name, param] : parameters) {
            if (param.requires_grad()) {
                parameters_.push_back(param);
            }
        }
    }
    
    // 梯度归零
    void zero_grad() {
        for (auto& param : parameters_) {
            param.zero_grad();
        }
    }
    
    // 参数更新（由子类实现）
    virtual void step() = 0;
    
    // getter和setter
    float get_learning_rate() const { return learning_rate_; }
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    
protected:
    std::vector<Variable> parameters_;  // 需要优化的参数
    float learning_rate_;              // 学习率
};
```

### SGD实现

随机梯度下降（SGD）优化器的实现：

```cpp
SGD::SGD(const std::unordered_map<std::string, Variable>& parameters, 
         float learning_rate,
         float momentum,
         float weight_decay)
    : Optimizer(parameters, learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    
    // 如果使用动量，初始化速度向量
    if (momentum_ > 0.0f) {
        for (const auto& param : parameters_) {
            auto shape = param.data().shape();
            velocity_.push_back(Tensor::zeros(shape));
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        
        // 获取数据和梯度
        auto data = param.data();
        auto grad = param.grad();
        
        // L2正则化（权重衰减）
        if (weight_decay_ > 0.0f) {
            // grad = grad + weight_decay * data
            Tensor weight_decay_grad = data * Tensor({1}, {weight_decay_}, false);
            grad = grad + weight_decay_grad;
        }
        
        // 使用动量
        if (momentum_ > 0.0f) {
            // v = momentum * v + grad
            velocity_[i] = velocity_[i] * Tensor({1}, {momentum_}, false) + grad;
            
            // data = data - learning_rate * v
            Tensor update = velocity_[i] * Tensor({1}, {learning_rate_}, false);
            param = Variable(data - update, param.requires_grad());
        } else {
            // 标准SGD: data = data - learning_rate * grad
            Tensor update = grad * Tensor({1}, {learning_rate_}, false);
            param = Variable(data - update, param.requires_grad());
        }
    }
}
```

### Adam实现

Adam优化器是一种自适应学习率的优化算法：

```cpp
Adam::Adam(const std::unordered_map<std::string, Variable>& parameters, 
          float learning_rate,
          float beta1,
          float beta2,
          float epsilon,
          float weight_decay)
    : Optimizer(parameters, learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      step_count_(0) {
    
    // 初始化动量和二阶矩向量
    for (const auto& param : parameters_) {
        auto shape = param.data().shape();
        m_.push_back(Tensor::zeros(shape));  // 一阶矩估计（动量）
        v_.push_back(Tensor::zeros(shape));  // 二阶矩估计（梯度平方的动量）
    }
}

void Adam::step() {
    step_count_++;
    
    // 计算偏置校正因子
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        
        auto data = param.data();
        auto grad = param.grad();
        
        // L2正则化
        if (weight_decay_ > 0.0f) {
            grad = grad + data * Tensor({1}, {weight_decay_}, false);
        }
        
        // 更新一阶矩估计
        m_[i] = m_[i] * Tensor({1}, {beta1_}, false) + grad * Tensor({1}, {1.0f - beta1_}, false);
        
        // 更新二阶矩估计
        auto grad_squared = grad * grad;
        v_[i] = v_[i] * Tensor({1}, {beta2_}, false) + grad_squared * Tensor({1}, {1.0f - beta2_}, false);
        
        // 计算偏置校正
        Tensor m_corrected = m_[i] * Tensor({1}, {1.0f / bias_correction1}, false);
        Tensor v_corrected = v_[i] * Tensor({1}, {1.0f / bias_correction2}, false);
        
        // 计算更新值
        std::vector<float> sqrt_v_data(v_corrected.numel());
        for (size_t j = 0; j < v_corrected.numel(); ++j) {
            sqrt_v_data[j] = std::sqrt(v_corrected[j]) + epsilon_;
        }
        
        Tensor sqrt_v_corrected(v_corrected.shape(), sqrt_v_data);
        
        // 更新参数：data = data - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        Tensor update = m_corrected / sqrt_v_corrected * Tensor({1}, {learning_rate_}, false);
        param = Variable(data - update, param.requires_grad());
    }
}
```

## Python 绑定

PyTorchCPP使用pybind11创建Python绑定，将C++类和函数暴露给Python。

### 模块初始化

```cpp
PYBIND11_MODULE(pytorchcpp, m) {
    m.doc() = "PyTorchCPP: A lightweight deep learning library in C++";
    
    // 创建子模块
    auto tensor_submodule = m.def_submodule("tensor", "Tensor operations");
    auto autograd_submodule = m.def_submodule("autograd", "Automatic differentiation");
    auto nn_submodule = m.def_submodule("nn", "Neural network modules");
    auto optim_submodule = m.def_submodule("optim", "Optimization algorithms");
    
    // 初始化各个模块的绑定
    init_tensor(m);
    init_autograd(m);
    init_nn(nn_submodule);
    init_optim(optim_submodule);
}
```

### 张量绑定

```cpp
void init_tensor(py::module_ &m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&, bool>(), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&, bool>(), 
             py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)
        
        // 工厂方法
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("requires_grad") = false)
        
        // 属性
        .def("shape", &Tensor::shape)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)
        .def("requires_grad", &Tensor::requires_grad)
        
        // 运算
        .def("matmul", &Tensor::matmul, py::arg("other"))
        .def("transpose", &Tensor::transpose, py::arg("dim0") = 0, py::arg("dim1") = 1)
        
        // 运算符重载
        .def("__add__", &Tensor::operator+, py::arg("other"))
        .def("__sub__", &Tensor::operator-, py::arg("other"))
        .def("__mul__", &Tensor::operator*, py::arg("other"))
        .def("__truediv__", &Tensor::operator/, py::arg("other"))
        
        // 字符串表示
        .def("__repr__", &Tensor::to_string)
        .def("__str__", &Tensor::to_string);
    
    // NumPy互操作
    m.def("from_numpy", [](py::array_t<float> array) {
        // 从NumPy数组创建Tensor
        // ...实现代码...
    });
    
    m.def("to_numpy", [](const Tensor &tensor) {
        // 将Tensor转换为NumPy数组
        // ...实现代码...
    });
}
```

### 神经网络模块绑定

```cpp
void init_nn(py::module_ &m) {
    // 绑定Module基类
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("__call__", &Module::operator())
        .def("parameters", &Module::parameters);
    
    // 绑定Linear类
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<size_t, size_t, bool>(), 
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true);
    
    // 绑定激活函数
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());
    
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());
    
    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>());
    
    // 绑定Sequential容器
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add);
    
    // 绑定损失函数
    py::class_<MSELoss, Module, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", static_cast<Variable (MSELoss::*)(const Variable&, const Variable&)>(&MSELoss::forward))
        .def("__call__", [](MSELoss& self, const Variable& input, const Variable& target) {
            return self.forward(input, target);
        });
}
```

注意使用`std::shared_ptr`来正确处理Python和C++之间的对象生命周期。

## 结论

PyTorchCPP实现了一个简化但功能完整的深度学习框架，包括张量操作、自动求导、神经网络模块和优化器。它采用了现代C++特性和设计模式，如PIMPL、智能指针和面向对象编程。Python绑定允许用户从Python中访问这些功能，实现了类似PyTorch的API。

虽然这是一个教学项目，没有针对性能进行优化，但它提供了对深度学习框架内部工作原理的宝贵见解。如需进一步改进，可以考虑添加GPU支持、并行计算、更多的层类型和优化器，以及性能优化。 
#include <pytorchcpp/nn.h>
#include <algorithm>
#include <cmath>

namespace pytorchcpp {
namespace nn {

// ReLU激活函数
ReLU::ReLU() = default;

Variable ReLU::forward(const Variable& input) {
    // ReLU: max(0, x)
    auto input_data = input.data();
    auto shape = input_data.shape();
    size_t size = input_data.numel();
    
    std::vector<float> result_data(size);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(0.0f, input_data[i]);
    }
    
    return Variable(Tensor(shape, result_data), input.requires_grad());
}

// Sigmoid激活函数
Sigmoid::Sigmoid() = default;

Variable Sigmoid::forward(const Variable& input) {
    // Sigmoid: 1 / (1 + exp(-x))
    auto input_data = input.data();
    auto shape = input_data.shape();
    size_t size = input_data.numel();
    
    std::vector<float> result_data(size);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }
    
    return Variable(Tensor(shape, result_data), input.requires_grad());
}

// Tanh激活函数
Tanh::Tanh() = default;

Variable Tanh::forward(const Variable& input) {
    // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    auto input_data = input.data();
    auto shape = input_data.shape();
    size_t size = input_data.numel();
    
    std::vector<float> result_data(size);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::tanh(input_data[i]);
    }
    
    return Variable(Tensor(shape, result_data), input.requires_grad());
}

} // namespace nn
} // namespace pytorchcpp 
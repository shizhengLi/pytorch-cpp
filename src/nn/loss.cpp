#include <pytorchcpp/nn.h>
#include <cmath>
#include <stdexcept>

namespace pytorchcpp {
namespace nn {

// MSE损失函数
MSELoss::MSELoss() = default;

Variable MSELoss::forward(const Variable& input, const Variable& target) {
    // 检查形状
    auto input_shape = input.data().shape();
    auto target_shape = target.data().shape();
    
    if (input_shape != target_shape) {
        throw std::invalid_argument("Input and target shapes must match for MSE loss");
    }
    
    // 计算平方差
    auto diff = input - target;
    auto squared_diff = diff * diff;
    
    // 计算平均值
    auto loss = squared_diff.data().mean();
    return Variable(loss, input.requires_grad() || target.requires_grad());
}

Variable MSELoss::forward(const Variable& input) {
    throw std::runtime_error("MSELoss requires both input and target");
}

// 交叉熵损失函数
CrossEntropyLoss::CrossEntropyLoss() = default;

Variable CrossEntropyLoss::forward(const Variable& input, const Variable& target) {
    // 检查输入是否为2D (batch_size x num_classes)
    auto input_shape = input.data().shape();
    if (input_shape.size() != 2) {
        throw std::invalid_argument("Expected 2D input for CrossEntropyLoss (batch_size x num_classes)");
    }
    
    // 检查目标是否为1D (batch_size)
    auto target_shape = target.data().shape();
    if (target_shape.size() != 1 || target_shape[0] != input_shape[0]) {
        throw std::invalid_argument("Target shape should be (batch_size) for CrossEntropyLoss");
    }
    
    // 实现简化版的交叉熵损失
    // 首先计算每个样本的softmax概率
    auto input_data = input.data();
    auto target_data = target.data();
    
    size_t batch_size = input_shape[0];
    size_t num_classes = input_shape[1];
    
    float loss_sum = 0.0f;
    
    // 对每个样本计算交叉熵
    for (size_t i = 0; i < batch_size; ++i) {
        // 计算softmax
        float max_val = input_data.at({i, 0});
        for (size_t j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, input_data.at({i, j}));
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; ++j) {
            sum_exp += std::exp(input_data.at({i, j}) - max_val);
        }
        
        // 目标类别索引
        size_t target_idx = static_cast<size_t>(target_data[i]);
        if (target_idx >= num_classes) {
            throw std::invalid_argument("Target index out of range");
        }
        
        // 计算交叉熵: -log(softmax(x)[target])
        loss_sum += -(input_data.at({i, target_idx}) - max_val - std::log(sum_exp));
    }
    
    // 返回平均损失
    float loss_avg = loss_sum / batch_size;
    return Variable(Tensor({1}, {loss_avg}, false), input.requires_grad() || target.requires_grad());
}

Variable CrossEntropyLoss::forward(const Variable& input) {
    throw std::runtime_error("CrossEntropyLoss requires both input and target");
}

} // namespace nn
} // namespace pytorchcpp 
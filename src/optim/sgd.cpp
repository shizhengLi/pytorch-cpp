#include <pytorchcpp/optim.h>
#include <iostream>

namespace pytorchcpp {
namespace optim {

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
        
        // 只更新需要梯度的参数
        if (!param.requires_grad()) {
            continue;
        }
        
        // 获取数据和梯度
        auto data = param.data();
        auto grad = param.grad();
        
        // 检查梯度是否为空或形状不匹配
        if (grad.numel() == 0) {
            continue;
        }
        
        auto data_shape = data.shape();
        auto grad_shape = grad.shape();
        
        if (data_shape != grad_shape) {
            continue;
        }
        
        // 应用权重衰减 (L2 正则化)
        std::vector<float> grad_data(data.numel());
        for (size_t j = 0; j < data.numel(); ++j) {
            grad_data[j] = grad[j];
            if (weight_decay_ > 0.0f) {
                grad_data[j] += weight_decay_ * data[j];
            }
        }
        
        std::vector<float> new_data(data.numel());
        
        // 使用动量的SGD
        if (momentum_ > 0.0f) {
            // 确保velocity_有足够的元素
            if (i >= velocity_.size()) {
                velocity_.push_back(Tensor::zeros(data_shape));
            }
            
            // 更新动量 (v = momentum * v + grad)
            for (size_t j = 0; j < data.numel(); ++j) {
                velocity_[i][j] = momentum_ * velocity_[i][j] + grad_data[j];
                new_data[j] = data[j] - learning_rate_ * velocity_[i][j];
            }
        } else {
            // 标准SGD (param = param - lr * grad)
            for (size_t j = 0; j < data.numel(); ++j) {
                new_data[j] = data[j] - learning_rate_ * grad_data[j];
            }
        }
        
        // 更新参数
        param = Variable(Tensor(data_shape, new_data), param.requires_grad());
    }
}

} // namespace optim
} // namespace pytorchcpp 
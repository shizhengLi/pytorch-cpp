#include <pytorchcpp/optim.h>

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
        
        // 权重衰减 (L2 正则化)
        if (weight_decay_ > 0.0f) {
            Tensor weight_decay_grad = data * Tensor({1}, {weight_decay_}, false);
            grad = grad + weight_decay_grad;
        }
        
        // 使用动量的SGD
        if (momentum_ > 0.0f) {
            velocity_[i] = velocity_[i] * Tensor({1}, {momentum_}, false) + grad;
            
            // 更新参数: w = w - lr * v
            Tensor update = velocity_[i] * Tensor({1}, {learning_rate_}, false);
            param = Variable(data - update, param.requires_grad());
        } else {
            // 标准SGD: w = w - lr * grad
            Tensor update = grad * Tensor({1}, {learning_rate_}, false);
            param = Variable(data - update, param.requires_grad());
        }
    }
}

} // namespace optim
} // namespace pytorchcpp 
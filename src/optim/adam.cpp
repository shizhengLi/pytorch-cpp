#include <pytorchcpp/optim.h>
#include <cmath>

namespace pytorchcpp {
namespace optim {

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
    
    // 初始化动量向量和速度向量
    for (const auto& param : parameters_) {
        auto shape = param.data().shape();
        m_.push_back(Tensor::zeros(shape));  // 一阶矩
        v_.push_back(Tensor::zeros(shape));  // 二阶矩
    }
}

void Adam::step() {
    step_count_++;
    
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
    
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
        
        // 更新偏置校正后的一阶矩估计
        m_[i] = m_[i] * Tensor({1}, {beta1_}, false) + grad * Tensor({1}, {1.0f - beta1_}, false);
        
        // 更新偏置校正后的二阶矩估计
        auto grad_squared = grad * grad;
        v_[i] = v_[i] * Tensor({1}, {beta2_}, false) + grad_squared * Tensor({1}, {1.0f - beta2_}, false);
        
        // 计算偏置校正
        Tensor m_corrected = m_[i] * Tensor({1}, {1.0f / bias_correction1}, false);
        Tensor v_corrected = v_[i] * Tensor({1}, {1.0f / bias_correction2}, false);
        
        // 准备更新
        std::vector<float> sqrt_v_data(v_corrected.numel());
        for (size_t j = 0; j < v_corrected.numel(); ++j) {
            sqrt_v_data[j] = std::sqrt(v_corrected[j]) + epsilon_;
        }
        
        Tensor sqrt_v_corrected(v_corrected.shape(), sqrt_v_data);
        Tensor update = m_corrected / sqrt_v_corrected * Tensor({1}, {learning_rate_}, false);
        
        // 更新参数
        param = Variable(data - update, param.requires_grad());
    }
}

} // namespace optim
} // namespace pytorchcpp 
#include <pytorchcpp/optim.h>
#include <cmath>
#include <iostream>

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
    
    // 计算偏置修正因子
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
        
        // 检查梯度是否为空或形状不匹配
        if (grad.numel() == 0) {
            continue;
        }
        
        auto data_shape = data.shape();
        auto grad_shape = grad.shape();
        
        if (data_shape != grad_shape) {
            continue;
        }
        
        // 确保动量和速度向量有正确的形状
        if (i >= m_.size()) {
            m_.push_back(Tensor::zeros(data_shape));
        }
        
        if (i >= v_.size()) {
            v_.push_back(Tensor::zeros(data_shape));
        }
        
        size_t numel = data.numel();
        std::vector<float> grad_data(numel);
        
        // 应用权重衰减 (L2 正则化)
        if (weight_decay_ > 0.0f) {
            for (size_t j = 0; j < numel; ++j) {
                grad_data[j] = grad[j] + weight_decay_ * data[j];
            }
        } else {
            for (size_t j = 0; j < numel; ++j) {
                grad_data[j] = grad[j];
            }
        }
        
        // 更新一阶矩估计 (移动平均)
        for (size_t j = 0; j < numel; ++j) {
            m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad_data[j];
        }
        
        // 更新二阶矩估计 (移动平均)
        for (size_t j = 0; j < numel; ++j) {
            float grad_squared = grad_data[j] * grad_data[j];
            v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad_squared;
        }
        
        // 计算偏置校正
        std::vector<float> new_data(numel);
        
        for (size_t j = 0; j < numel; ++j) {
            // 应用偏置校正
            float m_corrected = m_[i][j] / bias_correction1;
            float v_corrected = v_[i][j] / bias_correction2;
            
            // 更新参数: θ = θ - α * m / (√v + ε)
            float update = learning_rate_ * m_corrected / (std::sqrt(v_corrected) + epsilon_);
            new_data[j] = data[j] - update;
        }
        
        // 更新参数
        param = Variable(Tensor(data_shape, new_data), param.requires_grad());
    }
}

} // namespace optim
} // namespace pytorchcpp 
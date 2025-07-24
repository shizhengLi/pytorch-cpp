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
    std::cout << "Adam::step - 开始更新参数..." << std::endl;
    
    step_count_++;
    
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
    
    std::cout << "  步数: " << step_count_ << std::endl;
    std::cout << "  偏置校正1: " << bias_correction1 << std::endl;
    std::cout << "  偏置校正2: " << bias_correction2 << std::endl;
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        
        // 只更新需要梯度的参数
        if (!param.requires_grad()) {
            std::cout << "  参数 " << i << " 不需要梯度，跳过" << std::endl;
            continue;
        }
        
        // 获取数据和梯度
        auto data = param.data();
        auto grad = param.grad();
        
        auto data_shape = data.shape();
        auto grad_shape = grad.shape();
        
        std::cout << "  参数 " << i << " - 数据形状: [";
        for (size_t j = 0; j < data_shape.size(); ++j) {
            std::cout << data_shape[j];
            if (j < data_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  参数 " << i << " - 梯度形状: [";
        for (size_t j = 0; j < grad_shape.size(); ++j) {
            std::cout << grad_shape[j];
            if (j < grad_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 检查梯度是否为空或形状不匹配
        if (grad.numel() == 0) {
            std::cout << "  警告：梯度为空，跳过此参数" << std::endl;
            continue;
        }
        
        if (data_shape != grad_shape) {
            std::cout << "  警告：数据和梯度形状不匹配" << std::endl;
            continue;
        }
        
        // 确保动量和速度向量有正确的形状
        if (i >= m_.size()) {
            std::cout << "  初始化一阶矩向量 " << i << std::endl;
            m_.push_back(Tensor::zeros(data_shape));
        }
        
        if (i >= v_.size()) {
            std::cout << "  初始化二阶矩向量 " << i << std::endl;
            v_.push_back(Tensor::zeros(data_shape));
        }
        
        size_t numel = data.numel();
        std::vector<float> grad_data(numel);
        
        // 权重衰减 (L2 正则化)
        if (weight_decay_ > 0.0f) {
            std::cout << "  应用权重衰减: " << weight_decay_ << std::endl;
            
            for (size_t j = 0; j < numel; ++j) {
                float weight_decay_grad = data[j] * weight_decay_;
                grad_data[j] = grad[j] + weight_decay_grad;
            }
        } else {
            for (size_t j = 0; j < numel; ++j) {
                grad_data[j] = grad[j];
            }
        }
        
        // 更新一阶矩估计
        std::cout << "  更新一阶矩估计" << std::endl;
        for (size_t j = 0; j < numel; ++j) {
            m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad_data[j];
        }
        
        // 更新二阶矩估计
        std::cout << "  更新二阶矩估计" << std::endl;
        for (size_t j = 0; j < numel; ++j) {
            float grad_squared = grad_data[j] * grad_data[j];
            v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad_squared;
        }
        
        // 计算偏置校正
        std::cout << "  计算偏置校正" << std::endl;
        std::vector<float> m_corrected(numel);
        std::vector<float> v_corrected(numel);
        
        for (size_t j = 0; j < numel; ++j) {
            m_corrected[j] = m_[i][j] / bias_correction1;
            v_corrected[j] = v_[i][j] / bias_correction2;
        }
        
        // 计算更新值
        std::cout << "  计算更新值" << std::endl;
        std::vector<float> new_data(numel);
        
        for (size_t j = 0; j < numel; ++j) {
            float update = learning_rate_ * m_corrected[j] / (std::sqrt(v_corrected[j]) + epsilon_);
            new_data[j] = data[j] - update;
        }
        
        // 更新参数
        std::cout << "  更新参数 " << i << std::endl;
        param = Variable(Tensor(data_shape, new_data), param.requires_grad());
    }
    
    std::cout << "Adam::step - 参数更新完成" << std::endl;
}

} // namespace optim
} // namespace pytorchcpp 
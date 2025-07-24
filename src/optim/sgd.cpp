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
    std::cout << "SGD::step - 开始更新参数..." << std::endl;
    
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
        
        std::cout << "  参数 " << i << " - 数据形状: [";
        auto data_shape = data.shape();
        for (size_t j = 0; j < data_shape.size(); ++j) {
            std::cout << data_shape[j];
            if (j < data_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  参数 " << i << " - 梯度形状: [";
        auto grad_shape = grad.shape();
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
        
        // 安全地逐元素执行操作
        std::vector<float> update_data(data.numel());
        
        // 权重衰减 (L2 正则化)
        if (weight_decay_ > 0.0f) {
            std::cout << "  应用权重衰减: " << weight_decay_ << std::endl;
            
            for (size_t j = 0; j < data.numel(); ++j) {
                float weight_decay_grad = data[j] * weight_decay_;
                update_data[j] = grad[j] + weight_decay_grad;
            }
            
            // 用更新后的梯度替换原始梯度
            grad = Tensor(grad_shape, update_data);
            
            // 清空update_data以便重用
            std::fill(update_data.begin(), update_data.end(), 0.0f);
        }
        
        // 使用动量的SGD
        if (momentum_ > 0.0f) {
            std::cout << "  应用动量: " << momentum_ << std::endl;
            
            // 确保velocity_有足够的元素
            if (i >= velocity_.size()) {
                std::cout << "  初始化速度向量 " << i << std::endl;
                velocity_.push_back(Tensor::zeros(data_shape));
            }
            
            // 更新动量
            for (size_t j = 0; j < data.numel(); ++j) {
                velocity_[i][j] = velocity_[i][j] * momentum_ + grad[j];
                update_data[j] = velocity_[i][j] * learning_rate_;
            }
        } else {
            std::cout << "  应用标准SGD" << std::endl;
            // 标准SGD
            for (size_t j = 0; j < data.numel(); ++j) {
                update_data[j] = grad[j] * learning_rate_;
            }
        }
        
        // 计算更新后的参数
        std::vector<float> new_data(data.numel());
        for (size_t j = 0; j < data.numel(); ++j) {
            new_data[j] = data[j] - update_data[j];
        }
        
        // 更新参数
        param = Variable(Tensor(data_shape, new_data), param.requires_grad());
        std::cout << "  参数 " << i << " 已更新" << std::endl;
    }
    
    std::cout << "SGD::step - 参数更新完成" << std::endl;
}

} // namespace optim
} // namespace pytorchcpp 
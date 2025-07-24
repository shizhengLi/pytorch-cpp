#include <pytorchcpp/nn.h>
#include <cmath>
#include <random>

namespace pytorchcpp {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
    
    // 权重初始化 (He初始化)
    float stdv = 1.0f / std::sqrt(static_cast<float>(in_features));
    
    // 创建随机分布
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-stdv, stdv);
    
    // 创建并初始化权重
    std::vector<size_t> weight_shape = {out_features, in_features};
    std::vector<float> weight_data(out_features * in_features);
    for (auto& val : weight_data) {
        val = dist(gen);
    }
    weight_ = Variable(Tensor(weight_shape, weight_data), true);
    
    // 创建并初始化偏置 (如果需要)
    if (has_bias_) {
        std::vector<size_t> bias_shape = {out_features};
        std::vector<float> bias_data(out_features);
        for (auto& val : bias_data) {
            val = dist(gen);
        }
        bias_ = Variable(Tensor(bias_shape, bias_data), true);
    }
    
    // 注册参数
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

Variable Linear::forward(const Variable& input) {
    // y = x * W^T + b
    
    // 检查输入形状
    auto input_shape = input.data().shape();
    if (input_shape.back() != in_features_) {
        throw std::invalid_argument("Expected input features: " + std::to_string(in_features_) + 
                                   ", got: " + std::to_string(input_shape.back()));
    }
    
    // 计算 x * W^T
    auto output = input.matmul(Variable(weight_.data().transpose()));
    
    // 如果有偏置，则添加
    if (has_bias_) {
        // 修复：确保偏置的形状与输出匹配，通过广播机制添加偏置
        // 获取输出形状以确定批量大小
        auto output_shape = output.data().shape();
        
        if (output_shape.size() == 2) {
            // 批量输入情况：[batch_size, out_features]
            size_t batch_size = output_shape[0];
            
            // 创建与输出形状匹配的偏置张量，通过复制偏置到每个批次样本
            std::vector<float> expanded_bias_data;
            expanded_bias_data.reserve(batch_size * out_features_);
            
            // 为每个批次样本复制偏置
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features_; ++j) {
                    expanded_bias_data.push_back(bias_.data()[j]);
                }
            }
            
            // 创建广播后的偏置变量
            Tensor expanded_bias(output_shape, expanded_bias_data, bias_.requires_grad());
            Variable bias_broadcasted(expanded_bias, bias_.requires_grad());
            
            // 添加广播后的偏置
            output = output + bias_broadcasted;
        } else {
            // 单个样本情况：直接添加偏置
            output = output + bias_;
        }
    }
    
    return output;
}

} // namespace nn
} // namespace pytorchcpp 
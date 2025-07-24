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
        output = output + bias_;
    }
    
    return output;
}

} // namespace nn
} // namespace pytorchcpp 
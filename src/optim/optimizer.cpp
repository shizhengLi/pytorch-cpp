#include <pytorchcpp/optim.h>

namespace pytorchcpp {
namespace optim {

Optimizer::Optimizer(const std::unordered_map<std::string, Variable>& parameters, float learning_rate)
    : learning_rate_(learning_rate) {
    
    // 提取并保存所有参数
    size_t idx = 0;
    for (const auto& [name, param] : parameters) {
        parameters_.push_back(param);
        param_indices_[name] = idx++;
    }
}

void Optimizer::zero_grad() {
    for (auto& param : parameters_) {
        param.zero_grad();
    }
}

void Optimizer::set_learning_rate(float lr) {
    learning_rate_ = lr;
}

float Optimizer::get_learning_rate() const {
    return learning_rate_;
}

} // namespace optim
} // namespace pytorchcpp 
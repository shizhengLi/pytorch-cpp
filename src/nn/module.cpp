#include <pytorchcpp/nn.h>

namespace pytorchcpp {
namespace nn {

Module::Module() = default;

Variable Module::operator()(const Variable& input) {
    return forward(input);
}

std::unordered_map<std::string, Variable> Module::parameters() const {
    return params_;
}

void Module::register_parameter(const std::string& name, const Variable& param) {
    params_[name] = param;
}

} // namespace nn
} // namespace pytorchcpp
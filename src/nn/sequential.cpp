#include <pytorchcpp/nn.h>

namespace pytorchcpp {
namespace nn {

Sequential::Sequential() = default;

void Sequential::add(std::shared_ptr<Module> module) {
    modules_.push_back(module);
}

Variable Sequential::forward(const Variable& input) {
    Variable output = input;
    
    // 按顺序执行每个模块
    for (const auto& module : modules_) {
        output = module->forward(output);
    }
    
    return output;
}

std::unordered_map<std::string, Variable> Sequential::parameters() const {
    std::unordered_map<std::string, Variable> all_params;
    
    // 收集所有子模块的参数
    for (size_t i = 0; i < modules_.size(); ++i) {
        auto module_params = modules_[i]->parameters();
        
        // 加入前缀避免参数名冲突
        for (const auto& [name, param] : module_params) {
            all_params[std::to_string(i) + "." + name] = param;
        }
    }
    
    return all_params;
}

} // namespace nn
} // namespace pytorchcpp 
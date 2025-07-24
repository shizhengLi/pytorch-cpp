#pragma once

#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace pytorchcpp {
namespace optim {

/**
 * @brief 优化器基类
 */
class Optimizer {
public:
    Optimizer(const std::unordered_map<std::string, Variable>& parameters, float learning_rate = 0.01f);
    virtual ~Optimizer() = default;
    
    // 禁用拷贝
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    
    // 梯度归零
    void zero_grad();
    
    // 参数更新步骤
    virtual void step() = 0;
    
    // 设置学习率
    void set_learning_rate(float lr);
    float get_learning_rate() const;
    
protected:
    std::vector<Variable> parameters_;
    std::unordered_map<std::string, size_t> param_indices_;
    float learning_rate_;
};

/**
 * @brief 随机梯度下降优化器
 */
class SGD : public Optimizer {
public:
    SGD(const std::unordered_map<std::string, Variable>& parameters, 
        float learning_rate = 0.01f,
        float momentum = 0.0f,
        float weight_decay = 0.0f);
    
    void step() override;
    
private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;
};

/**
 * @brief Adam优化器
 */
class Adam : public Optimizer {
public:
    Adam(const std::unordered_map<std::string, Variable>& parameters, 
         float learning_rate = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float epsilon = 1e-8f,
         float weight_decay = 0.0f);
    
    void step() override;
    
private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    size_t step_count_;
    std::vector<Tensor> m_;  // 一阶矩估计
    std::vector<Tensor> v_;  // 二阶矩估计
};

} // namespace optim
} // namespace pytorchcpp 
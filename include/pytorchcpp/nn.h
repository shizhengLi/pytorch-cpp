#pragma once

#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace pytorchcpp {
namespace nn {

/**
 * @brief 神经网络模块基类
 */
class Module {
public:
    Module();
    virtual ~Module() = default;
    
    // 禁用拷贝
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    
    // 前向传播
    virtual Variable forward(const Variable& input) = 0;
    
    // 调用操作符，简化前向传播调用
    Variable operator()(const Variable& input);
    
    // 模块参数
    virtual std::unordered_map<std::string, Variable> parameters() const;
    
protected:
    std::unordered_map<std::string, Variable> params_;
    bool training_ = true;
    
    // 注册参数
    void register_parameter(const std::string& name, const Variable& param);
};

/**
 * @brief 线性层 (全连接层)
 */
class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    ~Linear() override = default;
    
    Variable forward(const Variable& input) override;
    
private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    
    // 权重和偏置
    Variable weight_;
    Variable bias_;
};

/**
 * @brief ReLU激活函数
 */
class ReLU : public Module {
public:
    ReLU();
    ~ReLU() override = default;
    
    Variable forward(const Variable& input) override;
    
private:
    bool inplace_ = false;
};

/**
 * @brief Sigmoid激活函数
 */
class Sigmoid : public Module {
public:
    Sigmoid();
    ~Sigmoid() override = default;
    
    Variable forward(const Variable& input) override;
};

/**
 * @brief Tanh激活函数
 */
class Tanh : public Module {
public:
    Tanh();
    ~Tanh() override = default;
    
    Variable forward(const Variable& input) override;
};

/**
 * @brief Sequential容器，按顺序执行模块
 */
class Sequential : public Module {
public:
    Sequential();
    ~Sequential() override = default;
    
    // 添加模块
    void add(std::shared_ptr<Module> module);
    
    // 前向传播
    Variable forward(const Variable& input) override;
    
    // 获取所有参数
    std::unordered_map<std::string, Variable> parameters() const override;
    
private:
    std::vector<std::shared_ptr<Module>> modules_;
};

/**
 * @brief MSE损失函数
 */
class MSELoss : public Module {
public:
    MSELoss();
    ~MSELoss() override = default;
    
    // 计算损失
    Variable forward(const Variable& input, const Variable& target);
    
    // 重写前向传播以处理单个输入
    Variable forward(const Variable& input) override;
};

/**
 * @brief 交叉熵损失函数
 */
class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss() override = default;
    
    // 计算损失
    Variable forward(const Variable& input, const Variable& target);
    
    // 重写前向传播以处理单个输入
    Variable forward(const Variable& input) override;
};

} // namespace nn
} // namespace pytorchcpp 
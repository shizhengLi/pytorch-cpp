#pragma once

#include <pytorchcpp/tensor.h>
#include <memory>
#include <vector>
#include <functional>

namespace pytorchcpp {

// 前向声明
class Function;

/**
 * @brief 变量类，实现自动求导的核心数据结构
 */
class Variable {
public:
    Variable();
    Variable(const Tensor& data, bool requires_grad = false);
    
    // 拷贝和移动构造/赋值
    Variable(const Variable& other);
    Variable(Variable&& other) noexcept;
    Variable& operator=(const Variable& other);
    Variable& operator=(Variable&& other) noexcept;
    
    // 析构函数
    ~Variable();
    
    // 访问器
    const Tensor& data() const;
    const Tensor& grad() const;
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    
    // 梯度操作
    void zero_grad();
    void backward(const Tensor& grad = Tensor());
    void set_grad(const Tensor& grad);  // 添加设置梯度方法
    
    // 运算操作
    Variable add(const Variable& other) const;
    Variable sub(const Variable& other) const;
    Variable mul(const Variable& other) const;
    Variable div(const Variable& other) const;
    Variable matmul(const Variable& other) const;
    
    // 运算符重载
    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;
    
private:
    struct Impl;
    std::shared_ptr<Impl> pImpl;  // 修改为shared_ptr以共享实现
    
    // 用于Function访问私有成员
    friend class Function;
};

/**
 * @brief 函数类，实现自动求导的计算图节点
 */
class Function {
public:
    Function();
    virtual ~Function() = default;
    
    // 禁用拷贝和赋值
    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;
    
    // 前向传播
    virtual Variable forward(const std::vector<Variable>& inputs) = 0;
    
    // 反向传播
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
    
    // 获取输入变量
    const std::vector<Variable>& get_inputs() const { return inputs; }
    
protected:
    std::vector<Variable> inputs;
    Variable output;
    
    // 保存输入输出的引用
    void save_for_backward(const std::vector<Variable>& inputs, const Variable& output);
};

// 基本操作的函数类
class AddFunction : public Function {
public:
    Variable forward(const std::vector<Variable>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class MulFunction : public Function {
public:
    Variable forward(const std::vector<Variable>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class MatmulFunction : public Function {
public:
    Variable forward(const std::vector<Variable>& inputs) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

} // namespace pytorchcpp 
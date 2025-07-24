#include <pytorchcpp/autograd.h>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace pytorchcpp {

// Variable的PIMPL实现
struct Variable::Impl {
    Tensor data;
    Tensor grad;
    bool requires_grad;
    std::shared_ptr<Function> grad_fn;
    
    Impl() : requires_grad(false) {}
    
    Impl(const Tensor& data, bool requires_grad)
        : data(data), requires_grad(requires_grad) {
        if (requires_grad) {
            // 创建与数据形状相同但全为0的梯度
            grad = Tensor::zeros(data.shape());
        }
    }
};

// 构造函数
Variable::Variable() : pImpl(std::make_unique<Impl>()) {}

Variable::Variable(const Tensor& data, bool requires_grad)
    : pImpl(std::make_unique<Impl>(data, requires_grad)) {}

// 拷贝和移动构造/赋值
Variable::Variable(const Variable& other) : pImpl(std::make_unique<Impl>(*other.pImpl)) {}

Variable::Variable(Variable&& other) noexcept = default;

Variable& Variable::operator=(const Variable& other) {
    if (this != &other) {
        *pImpl = *other.pImpl;
    }
    return *this;
}

Variable& Variable::operator=(Variable&& other) noexcept = default;

// 析构函数
Variable::~Variable() = default;

// 访问器
const Tensor& Variable::data() const {
    return pImpl->data;
}

const Tensor& Variable::grad() const {
    return pImpl->grad;
}

bool Variable::requires_grad() const {
    return pImpl->requires_grad;
}

void Variable::set_requires_grad(bool requires_grad) {
    pImpl->requires_grad = requires_grad;
    if (requires_grad && pImpl->grad.numel() == 0) {
        pImpl->grad = Tensor::zeros(pImpl->data.shape());
    }
}

// 梯度操作
void Variable::zero_grad() {
    if (pImpl->requires_grad) {
        pImpl->grad = Tensor::zeros(pImpl->data.shape());
    }
}

void Variable::backward(const Tensor& grad) {
    if (!pImpl->requires_grad) {
        throw std::runtime_error("Cannot call backward on a variable that doesn't require gradient");
    }
    
    // 如果未提供梯度，则假设为全1
    Tensor gradient = grad;
    if (grad.numel() == 0) {
        gradient = Tensor::ones({1}); // 标量梯度默认为1
        if (pImpl->data.numel() > 1) {
            // 非标量梯度需要形状匹配
            gradient = Tensor::ones(pImpl->data.shape());
        }
    }
    
    // 更新梯度
    pImpl->grad = pImpl->grad + gradient;
    
    // 如果有梯度函数，则继续反向传播
    if (pImpl->grad_fn) {
        auto grad_inputs = pImpl->grad_fn->backward(gradient);
        // 梯度传播由Function::backward负责
    }
}

void Variable::set_grad(const Tensor& grad) {
    if (!pImpl->requires_grad) {
        throw std::runtime_error("Cannot set grad for a variable that doesn't require gradient");
    }
    
    // 检查形状是否匹配
    if (pImpl->data.shape() != grad.shape()) {
        throw std::invalid_argument("Gradient shape must match data shape");
    }
    
    pImpl->grad = grad;
}

// 运算操作 - 通过Function实现
Variable Variable::add(const Variable& other) const {
    auto add_fn = std::make_shared<AddFunction>();
    auto result = add_fn->forward({*this, other});
    result.pImpl->grad_fn = add_fn;
    return result;
}

Variable Variable::sub(const Variable& other) const {
    // 减法可以通过加法和负数实现
    return add(Variable(other.data() * Tensor({1}, {-1.0f}, false), other.requires_grad()));
}

Variable Variable::mul(const Variable& other) const {
    auto mul_fn = std::make_shared<MulFunction>();
    auto result = mul_fn->forward({*this, other});
    result.pImpl->grad_fn = mul_fn;
    return result;
}

Variable Variable::div(const Variable& other) const {
    // 这里简化处理，除法可以通过乘法和倒数实现
    // 注意：实际实现应该有一个专门的DivFunction
    auto recip = Variable(Tensor::ones(other.data().shape()) / other.data(), other.requires_grad());
    return mul(recip);
}

Variable Variable::matmul(const Variable& other) const {
    auto matmul_fn = std::make_shared<MatmulFunction>();
    auto result = matmul_fn->forward({*this, other});
    result.pImpl->grad_fn = matmul_fn;
    return result;
}

// 运算符重载
Variable Variable::operator+(const Variable& other) const {
    return add(other);
}

Variable Variable::operator-(const Variable& other) const {
    return sub(other);
}

Variable Variable::operator*(const Variable& other) const {
    return mul(other);
}

Variable Variable::operator/(const Variable& other) const {
    return div(other);
}

} // namespace pytorchcpp 
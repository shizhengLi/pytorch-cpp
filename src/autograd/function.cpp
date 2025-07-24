#include <pytorchcpp/autograd.h>
#include <stdexcept>

namespace pytorchcpp {

Function::Function() = default;

void Function::save_for_backward(const std::vector<Variable>& inputs, const Variable& output) {
    this->inputs = inputs;
    this->output = output;
}

// AddFunction实现
Variable AddFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddFunction requires exactly 2 inputs");
    }
    
    Variable result(inputs[0].data() + inputs[1].data(), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> AddFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    // 加法的梯度简单地传回输入
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output;
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = grad_output;
    }
    
    return grad_inputs;
}

// MulFunction实现
Variable MulFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MulFunction requires exactly 2 inputs");
    }
    
    Variable result(inputs[0].data() * inputs[1].data(), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> MulFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    // 乘法的梯度：grad_a = grad_output * b, grad_b = grad_output * a
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output * inputs[1].data();
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = grad_output * inputs[0].data();
    }
    
    return grad_inputs;
}

// MatmulFunction实现
Variable MatmulFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatmulFunction requires exactly 2 inputs");
    }
    
    Variable result(inputs[0].data().matmul(inputs[1].data()), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> MatmulFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    // 矩阵乘法的梯度
    // grad_a = grad_output * b^T
    // grad_b = a^T * grad_output
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output.matmul(inputs[1].data().transpose());
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = inputs[0].data().transpose().matmul(grad_output);
    }
    
    return grad_inputs;
}

} // namespace pytorchcpp 
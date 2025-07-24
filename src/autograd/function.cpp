#include <pytorchcpp/autograd.h>
#include <stdexcept>
#include <iostream>  // 添加调试输出

namespace pytorchcpp {

Function::Function() = default;

void Function::save_for_backward(const std::vector<Variable>& inputs, const Variable& output) {
    // 直接保存输入和输出变量的引用
    this->inputs = inputs;
    this->output = output;
}

// AddFunction实现
Variable AddFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddFunction requires exactly 2 inputs");
    }
    
    std::cout << "AddFunction::forward - 输入要求梯度: " << inputs[0].requires_grad() << ", " << inputs[1].requires_grad() << std::endl;
    
    Variable result(inputs[0].data() + inputs[1].data(), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> AddFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    std::cout << "AddFunction::backward - 输入数量: " << inputs.size() << std::endl;
    
    // 加法的梯度简单地传回输入
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output;
        std::cout << "AddFunction::backward - 输入0梯度: " << grad_inputs[0] << std::endl;
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = grad_output;
        std::cout << "AddFunction::backward - 输入1梯度: " << grad_inputs[1] << std::endl;
    }
    
    return grad_inputs;
}

// MulFunction实现
Variable MulFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MulFunction requires exactly 2 inputs");
    }
    
    std::cout << "MulFunction::forward - 输入要求梯度: " << inputs[0].requires_grad() << ", " << inputs[1].requires_grad() << std::endl;
    
    Variable result(inputs[0].data() * inputs[1].data(), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> MulFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    std::cout << "MulFunction::backward - 输入数量: " << inputs.size() << std::endl;
    
    // 乘法的梯度：grad_a = grad_output * b, grad_b = grad_output * a
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output * inputs[1].data();
        std::cout << "MulFunction::backward - 输入0梯度 (grad_a = grad_output * b): " << grad_inputs[0] << std::endl;
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = grad_output * inputs[0].data();
        std::cout << "MulFunction::backward - 输入1梯度 (grad_b = grad_output * a): " << grad_inputs[1] << std::endl;
    }
    
    return grad_inputs;
}

// MatmulFunction实现
Variable MatmulFunction::forward(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatmulFunction requires exactly 2 inputs");
    }
    
    std::cout << "MatmulFunction::forward - 输入要求梯度: " << inputs[0].requires_grad() << ", " << inputs[1].requires_grad() << std::endl;
    
    Variable result(inputs[0].data().matmul(inputs[1].data()), 
                   inputs[0].requires_grad() || inputs[1].requires_grad());
    
    if (result.requires_grad()) {
        save_for_backward(inputs, result);
    }
    
    return result;
}

std::vector<Tensor> MatmulFunction::backward(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs(2);
    
    std::cout << "MatmulFunction::backward - 输入数量: " << inputs.size() << std::endl;
    
    // 矩阵乘法的梯度
    // grad_a = grad_output * b^T
    // grad_b = a^T * grad_output
    if (inputs[0].requires_grad()) {
        grad_inputs[0] = grad_output.matmul(inputs[1].data().transpose());
        std::cout << "MatmulFunction::backward - 输入0梯度 (grad_a = grad_output * b^T): " << grad_inputs[0] << std::endl;
    }
    
    if (inputs[1].requires_grad()) {
        grad_inputs[1] = inputs[0].data().transpose().matmul(grad_output);
        std::cout << "MatmulFunction::backward - 输入1梯度 (grad_b = a^T * grad_output): " << grad_inputs[1] << std::endl;
    }
    
    return grad_inputs;
}

} // namespace pytorchcpp 
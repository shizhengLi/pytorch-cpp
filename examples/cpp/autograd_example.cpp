#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <iostream>

int main() {
    using namespace pytorchcpp;
    
    std::cout << "PyTorchCPP 自动求导示例" << std::endl;
    std::cout << "-----------------------" << std::endl;
    
    // 创建变量
    Tensor x_data({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor y_data({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f}, true);
    
    Variable x(x_data, true);
    Variable y(y_data, true);
    
    std::cout << "变量 x 数据:" << std::endl;
    std::cout << x.data() << std::endl << std::endl;
    
    std::cout << "变量 y 数据:" << std::endl;
    std::cout << y.data() << std::endl << std::endl;
    
    // 前向计算
    // z = x * y + 2
    Variable z = x * y + Variable(Tensor({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f}, false));
    
    std::cout << "计算结果 z = x * y + 2:" << std::endl;
    std::cout << z.data() << std::endl << std::endl;
    
    // 反向传播
    z.backward();
    
    // 查看梯度
    std::cout << "x 的梯度:" << std::endl;
    std::cout << x.grad() << std::endl << std::endl;
    
    std::cout << "y 的梯度:" << std::endl;
    std::cout << y.grad() << std::endl << std::endl;
    
    // 更复杂的例子
    // 重置梯度
    x.zero_grad();
    y.zero_grad();
    
    // 计算 w = x^2 * y
    Variable w = x * x * y;
    
    std::cout << "计算结果 w = x^2 * y:" << std::endl;
    std::cout << w.data() << std::endl << std::endl;
    
    // 反向传播
    w.backward();
    
    // 查看梯度
    std::cout << "x 的梯度 (dw/dx = 2 * x * y):" << std::endl;
    std::cout << x.grad() << std::endl << std::endl;
    
    std::cout << "y 的梯度 (dw/dy = x^2):" << std::endl;
    std::cout << y.grad() << std::endl << std::endl;
    
    return 0;
} 
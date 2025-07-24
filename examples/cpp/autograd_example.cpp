#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <iostream>

int main() {
    using namespace pytorchcpp;
    
    std::cout << "PyTorchCPP 自动求导示例" << std::endl;
    std::cout << "-----------------------" << std::endl;
    
    // 简单示例: 标量变量
    std::cout << "简单标量示例:" << std::endl;
    Variable a(Tensor({1}, {2.0f}, false), true);  // a = 2
    Variable b(Tensor({1}, {3.0f}, false), true);  // b = 3
    
    // c = a * b
    Variable c = a * b;  // c = 2 * 3 = 6
    std::cout << "c = a * b = " << c.data() << std::endl;
    
    // 反向传播
    c.backward(Tensor({1}, {1.0f}, false));
    
    // 查看梯度: dc/da = b = 3, dc/db = a = 2
    std::cout << "a的梯度 (dc/da = b = 3): " << a.grad() << std::endl;
    std::cout << "b的梯度 (dc/db = a = 2): " << b.grad() << std::endl << std::endl;
    
    // 多维示例: 矩阵乘法
    std::cout << "矩阵乘法示例:" << std::endl;
    Tensor p_data({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, false);
    Tensor q_data({2, 1}, {1.0f, 2.0f}, false);
    
    Variable p(p_data, true);  // p = [[1, 2], [3, 4]]
    Variable q(q_data, true);  // q = [[1], [2]]
    
    // r = p @ q
    Variable r = p.matmul(q);  // r = [[5], [11]]
    std::cout << "r = p @ q = " << r.data() << std::endl;
    
    // 反向传播
    r.backward(Tensor({2, 1}, {1.0f, 1.0f}, false));
    
    // 查看梯度: dr/dp = q^T, dr/dq = p^T
    std::cout << "p的梯度 (dr/dp): " << p.grad() << std::endl;
    std::cout << "q的梯度 (dr/dq): " << q.grad() << std::endl << std::endl;
    
    // 复合函数示例: f(x, y) = (x * y) + (x * x * y)
    std::cout << "复合函数示例:" << std::endl;
    Variable x(Tensor({1}, {2.0f}, false), true);  // x = 2
    Variable y(Tensor({1}, {3.0f}, false), true);  // y = 3
    
    // 计算 f = x * y + x * x * y = 2*3 + 2*2*3 = 6 + 12 = 18
    Variable term1 = x * y;             // = 6
    Variable term2 = x * x * y;         // = 12
    Variable f = term1 + term2;         // = 18
    std::cout << "f(x,y) = x*y + x*x*y = " << f.data() << std::endl;
    
    // 反向传播
    f.backward(Tensor({1}, {1.0f}, false));
    
    // 查看梯度:
    // df/dx = y + 2*x*y = 3 + 2*2*3 = 3 + 12 = 15
    // df/dy = x + x*x = 2 + 4 = 6
    std::cout << "x的梯度 (df/dx = y + 2*x*y = 15): " << x.grad() << std::endl;
    std::cout << "y的梯度 (df/dy = x + x*x = 6): " << y.grad() << std::endl;
    
    return 0;
} 
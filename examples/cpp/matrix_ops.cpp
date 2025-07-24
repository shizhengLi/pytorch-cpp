#include <pytorchcpp/tensor.h>
#include <iostream>

int main() {
    // 使用命名空间
    using namespace pytorchcpp;
    
    std::cout << "PyTorchCPP 矩阵运算示例" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    // 创建张量
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::randn({3, 2});
    
    std::cout << "张量 a:" << std::endl;
    std::cout << a << std::endl << std::endl;
    
    std::cout << "张量 b:" << std::endl;
    std::cout << b << std::endl << std::endl;
    
    // 矩阵乘法
    std::cout << "矩阵乘法 a × b:" << std::endl;
    auto c = a.matmul(b);
    std::cout << c << std::endl << std::endl;
    
    // 转置
    std::cout << "转置 b^T:" << std::endl;
    auto b_t = b.transpose();
    std::cout << b_t << std::endl << std::endl;
    
    // 元素级操作
    std::cout << "张量加法 a + a:" << std::endl;
    std::cout << a + a << std::endl << std::endl;
    
    std::cout << "张量乘法 a * a (元素级):" << std::endl;
    std::cout << a * a << std::endl << std::endl;
    
    // 形状操作
    std::cout << "重塑张量 a 到 {3, 2}:" << std::endl;
    std::cout << a.reshape({3, 2}) << std::endl;
    
    return 0;
} 
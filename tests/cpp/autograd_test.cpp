#include <gtest/gtest.h>
#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <cmath>

using namespace pytorchcpp;

// 测试Variable的基本操作
TEST(AutogradTest, VariableBasics) {
    // 创建变量
    Tensor data({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Variable var(data, true);
    
    // 检查数据和梯度
    EXPECT_EQ(var.data().at({0, 0}), 1.0f);
    EXPECT_EQ(var.data().at({1, 1}), 4.0f);
    EXPECT_TRUE(var.requires_grad());
    
    // 初始梯度应该为零
    for (size_t i = 0; i < var.grad().numel(); ++i) {
        EXPECT_EQ(var.grad()[i], 0.0f);
    }
    
    // 测试零梯度
    var.zero_grad();
    for (size_t i = 0; i < var.grad().numel(); ++i) {
        EXPECT_EQ(var.grad()[i], 0.0f);
    }
}

// 测试简单的加法反向传播
TEST(AutogradTest, AddBackward) {
    // 创建变量
    Tensor x_data({1}, {2.0f}, false);
    Tensor y_data({1}, {3.0f}, false);
    
    Variable x(x_data, true);
    Variable y(y_data, true);
    
    // 计算 z = x + y
    Variable z = x + y;
    EXPECT_EQ(z.data()[0], 5.0f);
    
    // 反向传播
    z.backward();
    
    // 梯度应该都是1.0
    EXPECT_EQ(x.grad()[0], 1.0f);
    EXPECT_EQ(y.grad()[0], 1.0f);
}

// 测试简单的乘法反向传播
TEST(AutogradTest, MulBackward) {
    // 创建变量
    Tensor x_data({1}, {2.0f}, false);
    Tensor y_data({1}, {3.0f}, false);
    
    Variable x(x_data, true);
    Variable y(y_data, true);
    
    // 计算 z = x * y
    Variable z = x * y;
    EXPECT_EQ(z.data()[0], 6.0f);
    
    // 反向传播
    z.backward();
    
    // 验证梯度
    EXPECT_EQ(x.grad()[0], 3.0f);  // dz/dx = y = 3.0
    EXPECT_EQ(y.grad()[0], 2.0f);  // dz/dy = x = 2.0
}

// 测试复合函数
TEST(AutogradTest, CompositeFunction) {
    // 创建变量
    Tensor x_data({1}, {2.0f}, false);
    
    Variable x(x_data, true);
    
    // 计算 y = x^2 + x
    Variable y = x * x + x;
    EXPECT_EQ(y.data()[0], 6.0f);  // 2^2 + 2 = 6
    
    // 反向传播
    y.backward();
    
    // 验证梯度: dy/dx = 2x + 1
    EXPECT_EQ(x.grad()[0], 5.0f);  // 2*2 + 1 = 5
}

// 测试矩阵乘法
TEST(AutogradTest, MatmulBackward) {
    // 创建变量
    Tensor a_data({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b_data({2, 1}, {2.0f, 3.0f});
    
    Variable a(a_data, true);
    Variable b(b_data, true);
    
    // 计算 c = a @ b
    Variable c = a.matmul(b);
    EXPECT_EQ(c.data().at({0, 0}), 8.0f);   // [1, 2] @ [2, 3]^T = 1*2 + 2*3 = 8
    EXPECT_EQ(c.data().at({1, 0}), 18.0f);  // [3, 4] @ [2, 3]^T = 3*2 + 4*3 = 18
    
    // 反向传播
    c.backward();
    
    // 验证梯度
    // dc/da = b^T, dc/db = a^T
    EXPECT_EQ(a.grad().at({0, 0}), 2.0f);
    EXPECT_EQ(a.grad().at({0, 1}), 3.0f);
    EXPECT_EQ(a.grad().at({1, 0}), 2.0f);
    EXPECT_EQ(a.grad().at({1, 1}), 3.0f);
    
    EXPECT_EQ(b.grad().at({0, 0}), 4.0f);   // 1 + 3
    EXPECT_EQ(b.grad().at({1, 0}), 6.0f);   // 2 + 4
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
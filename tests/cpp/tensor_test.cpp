#include <gtest/gtest.h>
#include <pytorchcpp/tensor.h>

using namespace pytorchcpp;

TEST(TensorTest, Construction) {
    // 基本构造函数
    Tensor t1;
    EXPECT_EQ(t1.numel(), 0);
    
    // 形状构造
    Tensor t2({2, 3});
    EXPECT_EQ(t2.ndim(), 2);
    EXPECT_EQ(t2.shape()[0], 2);
    EXPECT_EQ(t2.shape()[1], 3);
    EXPECT_EQ(t2.numel(), 6);
    
    // 带数据构造
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t3({2, 2}, data);
    EXPECT_EQ(t3.at({0, 0}), 1.0f);
    EXPECT_EQ(t3.at({0, 1}), 2.0f);
    EXPECT_EQ(t3.at({1, 0}), 3.0f);
    EXPECT_EQ(t3.at({1, 1}), 4.0f);
}

TEST(TensorTest, FactoryMethods) {
    // 全零张量
    Tensor zeros = Tensor::zeros({2, 2});
    for (size_t i = 0; i < zeros.numel(); ++i) {
        EXPECT_EQ(zeros[i], 0.0f);
    }
    
    // 全一张量
    Tensor ones = Tensor::ones({2, 2});
    for (size_t i = 0; i < ones.numel(); ++i) {
        EXPECT_EQ(ones[i], 1.0f);
    }
}

TEST(TensorTest, BasicOperations) {
    // 准备测试张量
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {2.0f, 3.0f, 4.0f, 5.0f};
    Tensor t1({2, 2}, data1);
    Tensor t2({2, 2}, data2);
    
    // 加法
    Tensor sum = t1 + t2;
    EXPECT_EQ(sum.at({0, 0}), 3.0f);
    EXPECT_EQ(sum.at({0, 1}), 5.0f);
    EXPECT_EQ(sum.at({1, 0}), 7.0f);
    EXPECT_EQ(sum.at({1, 1}), 9.0f);
    
    // 减法
    Tensor diff = t2 - t1;
    EXPECT_EQ(diff.at({0, 0}), 1.0f);
    EXPECT_EQ(diff.at({0, 1}), 1.0f);
    EXPECT_EQ(diff.at({1, 0}), 1.0f);
    EXPECT_EQ(diff.at({1, 1}), 1.0f);
    
    // 乘法
    Tensor prod = t1 * t2;
    EXPECT_EQ(prod.at({0, 0}), 2.0f);
    EXPECT_EQ(prod.at({0, 1}), 6.0f);
    EXPECT_EQ(prod.at({1, 0}), 12.0f);
    EXPECT_EQ(prod.at({1, 1}), 20.0f);
    
    // 除法
    Tensor div = t2 / t1;
    EXPECT_FLOAT_EQ(div.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(div.at({0, 1}), 1.5f);
    EXPECT_FLOAT_EQ(div.at({1, 0}), 4.0f / 3.0f);
    EXPECT_FLOAT_EQ(div.at({1, 1}), 5.0f / 4.0f);
}

TEST(TensorTest, MatrixMultiplication) {
    // 准备测试张量
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor t1({2, 2}, data1);
    Tensor t2({2, 2}, data2);
    
    // 矩阵乘法
    Tensor result = t1.matmul(t2);
    EXPECT_EQ(result.at({0, 0}), 19.0f);  // 1*5 + 2*7
    EXPECT_EQ(result.at({0, 1}), 22.0f);  // 1*6 + 2*8
    EXPECT_EQ(result.at({1, 0}), 43.0f);  // 3*5 + 4*7
    EXPECT_EQ(result.at({1, 1}), 50.0f);  // 3*6 + 4*8
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
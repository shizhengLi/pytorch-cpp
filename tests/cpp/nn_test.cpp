#include <gtest/gtest.h>
#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <pytorchcpp/nn.h>
#include <memory>

using namespace pytorchcpp;
using namespace pytorchcpp::nn;

// 测试线性层
TEST(NNTest, Linear) {
    // 创建线性层
    Linear linear(2, 3);
    
    // 参数应该存在
    auto params = linear.parameters();
    EXPECT_TRUE(params.find("weight") != params.end());
    EXPECT_TRUE(params.find("bias") != params.end());
    
    // 权重应该是 3x2
    EXPECT_EQ(params["weight"].data().shape()[0], 3);
    EXPECT_EQ(params["weight"].data().shape()[1], 2);
    
    // 偏置应该是 3
    EXPECT_EQ(params["bias"].data().shape()[0], 3);
    
    // 所有参数都应该需要梯度
    EXPECT_TRUE(params["weight"].requires_grad());
    EXPECT_TRUE(params["bias"].requires_grad());
    
    // 前向传播
    Tensor input_data({1, 2}, {1.0f, 2.0f});
    Variable input(input_data);
    
    Variable output = linear.forward(input);
    
    // 输出应该是 1x3
    EXPECT_EQ(output.data().shape()[0], 1);
    EXPECT_EQ(output.data().shape()[1], 3);
}

// 测试激活函数
TEST(NNTest, Activations) {
    // 测试 ReLU
    ReLU relu;
    Tensor relu_input_data({2}, {-1.0f, 2.0f});
    Variable relu_input(relu_input_data);
    Variable relu_output = relu.forward(relu_input);
    
    EXPECT_EQ(relu_output.data()[0], 0.0f);
    EXPECT_EQ(relu_output.data()[1], 2.0f);
    
    // 测试 Sigmoid
    Sigmoid sigmoid;
    Tensor sigmoid_input_data({2}, {0.0f, 1.0f});
    Variable sigmoid_input(sigmoid_input_data);
    Variable sigmoid_output = sigmoid.forward(sigmoid_input);
    
    EXPECT_FLOAT_EQ(sigmoid_output.data()[0], 0.5f);
    EXPECT_FLOAT_EQ(sigmoid_output.data()[1], 1.0f / (1.0f + std::exp(-1.0f)));
    
    // 测试 Tanh
    Tanh tanh;
    Tensor tanh_input_data({2}, {0.0f, 1.0f});
    Variable tanh_input(tanh_input_data);
    Variable tanh_output = tanh.forward(tanh_input);
    
    EXPECT_FLOAT_EQ(tanh_output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(tanh_output.data()[1], std::tanh(1.0f));
}

// 测试Sequential容器
TEST(NNTest, Sequential) {
    // 创建网络
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(2, 3));
    seq->add(std::make_shared<ReLU>());
    seq->add(std::make_shared<Linear>(3, 1));
    
    // 参数应该包括两个线性层的权重和偏置
    auto params = seq->parameters();
    EXPECT_EQ(params.size(), 4);  // 2个权重和2个偏置
    
    // 前向传播
    Tensor input_data({1, 2}, {1.0f, 2.0f});
    Variable input(input_data);
    
    Variable output = seq->forward(input);
    
    // 输出应该是 1x1
    EXPECT_EQ(output.data().shape()[0], 1);
    EXPECT_EQ(output.data().shape()[1], 1);
}

// 测试损失函数
TEST(NNTest, LossFunctions) {
    // 测试 MSE 损失
    MSELoss mse;
    Tensor pred_data({2, 1}, {1.0f, 2.0f});
    Tensor target_data({2, 1}, {2.0f, 2.0f});
    
    Variable pred(pred_data);
    Variable target(target_data);
    
    Variable mse_loss = mse.forward(pred, target);
    
    // MSE = ((1-2)^2 + (2-2)^2) / 2 = 0.5
    EXPECT_FLOAT_EQ(mse_loss.data()[0], 0.5f);
    
    // 测试交叉熵损失
    CrossEntropyLoss ce;
    Tensor ce_pred_data({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor ce_target_data({2}, {2.0f, 1.0f});  // 类别索引: 第一个样本是类别2，第二个是类别1
    
    Variable ce_pred(ce_pred_data);
    Variable ce_target(ce_target_data);
    
    // 注意：这只是简单验证交叉熵能够运行，不会精确计算结果
    Variable ce_loss = ce.forward(ce_pred, ce_target);
    EXPECT_TRUE(ce_loss.data().numel() > 0);
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
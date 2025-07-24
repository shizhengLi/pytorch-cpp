#include <gtest/gtest.h>
#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <pytorchcpp/nn.h>
#include <pytorchcpp/optim.h>
#include <memory>
#include <cmath>

using namespace pytorchcpp;
using namespace pytorchcpp::nn;
using namespace pytorchcpp::optim;

// 测试SGD优化器
TEST(OptimTest, SGD) {
    // 创建简单参数
    Tensor weight_data({2, 1}, {1.0f, 2.0f});
    Tensor bias_data({1}, {0.5f}, false);
    
    Variable weight(weight_data, true);
    Variable bias(bias_data, true);
    
    // 创建参数字典
    std::unordered_map<std::string, Variable> params = {
        {"weight", weight},
        {"bias", bias}
    };
    
    // 创建SGD优化器
    SGD optimizer(params, 0.1f);
    EXPECT_FLOAT_EQ(optimizer.get_learning_rate(), 0.1f);
    
    // 设置梯度
    weight.set_grad(Tensor({2, 1}, {1.0f, 2.0f}));
    bias.set_grad(Tensor({1}, {0.5f}, false));
    
    // 执行一步优化
    optimizer.step();
    
    // 验证参数更新
    // weight = weight - lr * grad
    EXPECT_FLOAT_EQ(weight.data()[0], 0.9f);  // 1.0 - 0.1 * 1.0
    EXPECT_FLOAT_EQ(weight.data()[1], 1.8f);  // 2.0 - 0.1 * 2.0
    EXPECT_FLOAT_EQ(bias.data()[0], 0.45f);   // 0.5 - 0.1 * 0.5
    
    // 测试梯度归零
    optimizer.zero_grad();
    EXPECT_FLOAT_EQ(weight.grad()[0], 0.0f);
    EXPECT_FLOAT_EQ(weight.grad()[1], 0.0f);
    EXPECT_FLOAT_EQ(bias.grad()[0], 0.0f);
}

// 测试SGD优化器 (带动量)
TEST(OptimTest, SGDWithMomentum) {
    // 创建简单参数
    Tensor weight_data({1}, {1.0f}, false);
    Variable weight(weight_data, true);
    
    // 创建参数字典
    std::unordered_map<std::string, Variable> params = {
        {"weight", weight}
    };
    
    // 创建带动量的SGD优化器
    float lr = 0.1f;
    float momentum = 0.9f;
    SGD optimizer(params, lr, momentum);
    
    // 执行多步优化，模拟恒定梯度
    weight.set_grad(Tensor({1}, {1.0f}, false));
    
    // 第一步: v = 0*0.9 + 1.0 = 1.0, w = 1.0 - 0.1*1.0 = 0.9
    optimizer.step();
    EXPECT_FLOAT_EQ(weight.data()[0], 0.9f);
    
    // 第二步: v = 1.0*0.9 + 1.0 = 1.9, w = 0.9 - 0.1*1.9 = 0.71
    optimizer.step();
    EXPECT_FLOAT_EQ(weight.data()[0], 0.71f);
    
    // 第三步: v = 1.9*0.9 + 1.0 = 2.71, w = 0.71 - 0.1*2.71 = 0.439
    optimizer.step();
    EXPECT_FLOAT_EQ(weight.data()[0], 0.439f);
}

// 测试Adam优化器
TEST(OptimTest, Adam) {
    // 创建简单参数
    Tensor weight_data({1}, {1.0f}, false);
    Variable weight(weight_data, true);
    
    // 创建参数字典
    std::unordered_map<std::string, Variable> params = {
        {"weight", weight}
    };
    
    // 创建Adam优化器
    float lr = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    Adam optimizer(params, lr, beta1, beta2, epsilon);
    
    // 设置梯度
    weight.set_grad(Tensor({1}, {1.0f}, false));
    
    // 执行一步优化
    optimizer.step();
    
    // 验证参数更新 (不精确计算具体数值，只验证优化器能运行)
    EXPECT_LT(weight.data()[0], 1.0f);
    
    // 测试梯度归零
    optimizer.zero_grad();
    EXPECT_FLOAT_EQ(weight.grad()[0], 0.0f);
}

// 测试线性回归训练
TEST(OptimTest, LinearRegressionTraining) {
    // 创建简单的线性回归模型 y = 2x + 1
    Tensor x_data({10, 1});
    Tensor y_data({10, 1});
    
    for (size_t i = 0; i < 10; ++i) {
        float x = static_cast<float>(i) / 10.0f;
        float y = 2.0f * x + 1.0f;
        x_data[i] = x;
        y_data[i] = y;
    }
    
    Variable x(x_data);
    Variable y(y_data);
    
    // 创建模型
    auto model = std::make_shared<Linear>(1, 1);
    
    // 初始化参数
    model->parameters()["weight"] = Variable(Tensor({1, 1}, {0.5f}, false), true);
    model->parameters()["bias"] = Variable(Tensor({1}, {0.0f}, false), true);
    
    // 创建优化器和损失函数
    SGD optimizer(model->parameters(), 0.1f);
    MSELoss criterion;
    
    // 训练多个epoch
    const int num_epochs = 100;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 前向传播
        Variable y_pred = model->forward(x);
        
        // 计算损失
        Variable loss = criterion.forward(y_pred, y);
        
        // 反向传播
        optimizer.zero_grad();
        loss.backward();
        
        // 更新参数
        optimizer.step();
    }
    
    // 检查模型是否学习到了正确的参数
    float weight = model->parameters()["weight"].data()[0];
    float bias = model->parameters()["bias"].data()[0];
    
    // 允许一定的误差
    EXPECT_NEAR(weight, 2.0f, 0.1f);
    EXPECT_NEAR(bias, 1.0f, 0.1f);
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
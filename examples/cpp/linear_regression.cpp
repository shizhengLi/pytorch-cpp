#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <pytorchcpp/nn.h>
#include <pytorchcpp/optim.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main() {
    using namespace pytorchcpp;
    
    std::cout << "PyTorchCPP 线性回归示例" << std::endl;
    std::cout << "----------------------" << std::endl;
    
    // 生成一些合成数据: y = 2*x + 1 + noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    const size_t num_samples = 100;
    const size_t input_dim = 1;
    
    std::vector<float> x_data(num_samples);
    std::vector<float> y_data(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        float x = static_cast<float>(i) / num_samples;
        x_data[i] = x;
        y_data[i] = 2.0f * x + 1.0f + noise(gen);
    }
    
    // 创建张量
    Tensor x_tensor({num_samples, input_dim}, x_data);
    Tensor y_tensor({num_samples, 1}, y_data);
    
    Variable x(x_tensor);
    Variable y(y_tensor);
    
    std::cout << "数据生成完成: y = 2*x + 1 + noise" << std::endl;
    std::cout << "样本数量: " << num_samples << std::endl;
    
    // 创建模型
    auto model = std::make_shared<nn::Linear>(input_dim, 1);
    
    // 创建损失函数
    nn::MSELoss criterion;
    
    // 创建优化器
    optim::SGD optimizer(model->parameters(), 0.1f);
    
    // 训练模型
    const size_t num_epochs = 100;
    
    std::cout << std::endl << "开始训练..." << std::endl;
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // 前向传播
        auto y_pred = model->forward(x);
        
        // 计算损失
        auto loss = criterion.forward(y_pred, y);
        
        // 反向传播
        optimizer.zero_grad();
        loss.backward();
        
        // 更新参数
        optimizer.step();
        
        // 每10个epoch打印一次损失
        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                      << ", Loss: " << loss.data()[0] << std::endl;
        }
    }
    
    std::cout << "训练完成!" << std::endl << std::endl;
    
    // 获取模型参数
    auto params = model->parameters();
    float weight = params["weight"].data().at({0, 0});
    float bias = params["bias"].data()[0];
    
    std::cout << "模型参数:" << std::endl;
    std::cout << "weight: " << weight << " (真实值: 2.0)" << std::endl;
    std::cout << "bias: " << bias << " (真实值: 1.0)" << std::endl;
    
    return 0;
} 
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
    std::cout << "x形状: [" << x_tensor.shape()[0] << ", " << x_tensor.shape()[1] << "]" << std::endl;
    std::cout << "y形状: [" << y_tensor.shape()[0] << ", " << y_tensor.shape()[1] << "]" << std::endl;
    
    // 创建模型
    auto model = std::make_shared<nn::Linear>(input_dim, 1);
    
    // 创建损失函数
    nn::MSELoss criterion;
    
    // 创建优化器
    optim::SGD optimizer(model->parameters(), 0.1f);
    
    // 训练模型
    const size_t num_epochs = 10; // 减少迭代次数，先测试是否能运行
    
    std::cout << std::endl << "开始训练..." << std::endl;
    
    try {
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            // 前向传播
            std::cout << "Epoch " << (epoch + 1) << ": 执行前向传播..." << std::endl;
            auto y_pred = model->forward(x);
            std::cout << "y_pred形状: ";
            auto y_pred_shape = y_pred.data().shape();
            std::cout << "[";
            for (size_t i = 0; i < y_pred_shape.size(); ++i) {
                std::cout << y_pred_shape[i];
                if (i < y_pred_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // 计算损失
            std::cout << "计算损失..." << std::endl;
            auto loss = criterion.forward(y_pred, y);
            
            // 反向传播
            std::cout << "执行反向传播..." << std::endl;
            optimizer.zero_grad();
            loss.backward();
            
            // 更新参数
            std::cout << "更新参数..." << std::endl;
            optimizer.step();
            
            // 打印损失
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                      << ", Loss: " << loss.data()[0] << std::endl << std::endl;
        }
        
        std::cout << "训练完成!" << std::endl << std::endl;
        
        // 获取模型参数
        auto params = model->parameters();
        float weight = params["weight"].data().at({0, 0});
        float bias = params["bias"].data()[0];
        
        std::cout << "模型参数:" << std::endl;
        std::cout << "weight: " << weight << " (真实值: 2.0)" << std::endl;
        std::cout << "bias: " << bias << " (真实值: 1.0)" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "发生错误: " << e.what() << std::endl;
        
        // 详细分析错误发生的地方
        std::cout << "尝试单步调试..." << std::endl;
        
        // 前向传播测试
        std::cout << "测试前向传播..." << std::endl;
        auto y_pred = model->forward(x);
        std::cout << "前向传播成功! y_pred形状: ";
        auto y_pred_shape = y_pred.data().shape();
        for (size_t i = 0; i < y_pred_shape.size(); ++i) {
            std::cout << y_pred_shape[i] << " ";
        }
        std::cout << std::endl;
        
        try {
            std::cout << "测试损失计算..." << std::endl;
            auto loss = criterion.forward(y_pred, y);
            std::cout << "损失计算成功! 损失值: " << loss.data()[0] << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "损失计算失败: " << e.what() << std::endl;
            std::cout << "y_pred形状: ";
            auto pred_shape = y_pred.data().shape();
            for (auto s : pred_shape) std::cout << s << " ";
            std::cout << std::endl;
            
            std::cout << "y形状: ";
            auto target_shape = y.data().shape();
            for (auto s : target_shape) std::cout << s << " ";
            std::cout << std::endl;
        }
    }
    
    return 0;
} 
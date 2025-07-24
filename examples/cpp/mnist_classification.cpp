#include <pytorchcpp/tensor.h>
#include <pytorchcpp/autograd.h>
#include <pytorchcpp/nn.h>
#include <pytorchcpp/optim.h>
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>

using namespace pytorchcpp;
using namespace pytorchcpp::nn;

// MNIST数据加载器 (模拟版本，实际应用需要读取MNIST文件)
class MNISTSimulator {
public:
    MNISTSimulator(size_t num_samples = 1000, size_t batch_size = 64)
        : num_samples_(num_samples), batch_size_(batch_size) {
        // 初始化随机数生成器
        std::random_device rd;
        gen_ = std::mt19937(rd());
        dist_ = std::uniform_int_distribution<>(0, 9);  // 0-9的数字

        // 生成模拟数据
        generate_data();
    }

    size_t num_batches() const {
        return (num_samples_ + batch_size_ - 1) / batch_size_;  // 向上取整
    }

    std::pair<Variable, Variable> get_batch(size_t batch_idx) {
        size_t start_idx = batch_idx * batch_size_;
        size_t end_idx = std::min(start_idx + batch_size_, num_samples_);
        size_t actual_batch_size = end_idx - start_idx;

        // 提取此批次的数据
        std::vector<float> batch_data(actual_batch_size * 784);  // 28x28 = 784
        std::vector<float> batch_labels(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t idx = start_idx + i;
            // 复制图像数据
            std::copy(images_[idx].begin(), images_[idx].end(), 
                     batch_data.begin() + i * 784);
            
            // 复制标签
            batch_labels[i] = static_cast<float>(labels_[idx]);
        }

        // 创建变量
        Tensor x_tensor({actual_batch_size, 784}, batch_data);
        Tensor y_tensor({actual_batch_size}, batch_labels);

        return {Variable(x_tensor), Variable(y_tensor)};
    }

private:
    void generate_data() {
        images_.resize(num_samples_);
        labels_.resize(num_samples_);

        // 为每个样本生成模拟数据
        for (size_t i = 0; i < num_samples_; ++i) {
            // 随机选择一个数字类别 (0-9)
            int digit = dist_(gen_);
            labels_[i] = digit;

            // 创建一个28x28全0图像
            std::vector<float> image(784, 0.0f);

            // 根据数字添加一些模式
            // 这里我们只是添加一些随机值来模拟数字的外观
            std::normal_distribution<float> pixel_dist(0.1f, 0.3f);

            // 中心位置
            int center_x = 14;
            int center_y = 14;
            int radius = 10;

            // 根据数字添加不同的模式
            switch (digit) {
                case 0: {
                    // 圆形
                    for (int y = 0; y < 28; ++y) {
                        for (int x = 0; x < 28; ++x) {
                            float dist = std::sqrt(std::pow(x - center_x, 2) + 
                                                  std::pow(y - center_y, 2));
                            if (dist < radius && dist > radius - 4) {
                                image[y * 28 + x] = 0.8f + pixel_dist(gen_);
                            }
                        }
                    }
                    break;
                }
                case 1: {
                    // 垂直线
                    for (int y = 5; y < 23; ++y) {
                        for (int x = center_x - 1; x <= center_x + 1; ++x) {
                            image[y * 28 + x] = 0.8f + pixel_dist(gen_);
                        }
                    }
                    break;
                }
                // 其他数字都用随机模式
                default: {
                    for (int y = center_y - radius/2; y < center_y + radius/2; ++y) {
                        for (int x = center_x - radius/2; x < center_x + radius/2; ++x) {
                            if (y >= 0 && y < 28 && x >= 0 && x < 28) {
                                float val = 0.5f + 0.5f * std::sin(digit * (x + y) / 10.0f);
                                image[y * 28 + x] = val + 0.2f * pixel_dist(gen_);
                            }
                        }
                    }
                }
            }

            images_[i] = image;
        }
    }

    size_t num_samples_;
    size_t batch_size_;
    std::vector<std::vector<float>> images_;
    std::vector<int> labels_;
    std::mt19937 gen_;
    std::uniform_int_distribution<> dist_;
};

// 简单的神经网络模型用于MNIST分类
class MNISTNet {
public:
    MNISTNet() {
        // 创建层
        fc1_ = std::make_shared<Linear>(784, 128);  // 输入层 -> 隐藏层
        relu1_ = std::make_shared<ReLU>();
        fc2_ = std::make_shared<Linear>(128, 64);   // 隐藏层 -> 隐藏层 
        relu2_ = std::make_shared<ReLU>();
        fc3_ = std::make_shared<Linear>(64, 10);    // 隐藏层 -> 输出层
    }

    Variable forward(const Variable& x) {
        auto h1 = fc1_->forward(x);
        auto a1 = relu1_->forward(h1);
        auto h2 = fc2_->forward(a1);
        auto a2 = relu2_->forward(h2);
        auto out = fc3_->forward(a2);
        return out;
    }

    std::unordered_map<std::string, Variable> parameters() {
        std::unordered_map<std::string, Variable> params;
        
        // 合并所有层的参数
        auto merge_params = [&params](const std::string& prefix,
                                     const std::unordered_map<std::string, Variable>& layer_params) {
            for (const auto& [name, param] : layer_params) {
                params[prefix + "." + name] = param;
            }
        };
        
        merge_params("fc1", fc1_->parameters());
        merge_params("fc2", fc2_->parameters());
        merge_params("fc3", fc3_->parameters());
        
        return params;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<ReLU> relu2_;
    std::shared_ptr<Linear> fc3_;
};

// 计算准确率
float compute_accuracy(const Variable& output, const Variable& target) {
    auto output_data = output.data();
    auto target_data = target.data();
    
    size_t batch_size = output_data.shape()[0];
    size_t num_correct = 0;
    
    for (size_t i = 0; i < batch_size; ++i) {
        // 找到最大值的索引
        size_t pred_idx = 0;
        float max_val = output_data.at({i, 0});
        
        for (size_t j = 1; j < 10; ++j) {
            float val = output_data.at({i, j});
            if (val > max_val) {
                max_val = val;
                pred_idx = j;
            }
        }
        
        // 检查预测是否正确
        if (pred_idx == static_cast<size_t>(target_data[i])) {
            num_correct++;
        }
    }
    
    return static_cast<float>(num_correct) / static_cast<float>(batch_size);
}

int main() {
    std::cout << "PyTorchCPP MNIST分类示例" << std::endl;
    std::cout << "-----------------------" << std::endl;
    
    // 设置随机种子
    srand(42);
    
    // 创建模拟MNIST数据
    const size_t batch_size = 32;
    const size_t num_samples = 1000;
    MNISTSimulator train_data(num_samples, batch_size);
    
    // 创建模型
    MNISTNet model;
    
    // 创建损失函数和优化器
    CrossEntropyLoss criterion;
    optim::Adam optimizer(model.parameters(), 0.01f);
    
    // 训练参数
    const size_t num_epochs = 5;
    
    std::cout << "开始训练..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 训练循环
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        float total_acc = 0.0f;
        
        size_t num_batches = train_data.num_batches();
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            // 获取一批数据
            auto [inputs, targets] = train_data.get_batch(batch);
            
            // 前向传播
            auto outputs = model.forward(inputs);
            auto loss = criterion.forward(outputs, targets);
            
            // 计算准确率
            float accuracy = compute_accuracy(outputs, targets);
            
            // 反向传播和优化
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            // 累计统计
            total_loss += loss.data()[0];
            total_acc += accuracy;
        }
        
        // 计算平均值
        float avg_loss = total_loss / num_batches;
        float avg_acc = total_acc / num_batches;
        
        std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], "
                  << "Loss: " << avg_loss << ", "
                  << "Accuracy: " << (avg_acc * 100.0f) << "%" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "训练完成! 用时: " << duration / 1000.0f << " 秒" << std::endl;
    
    // 测试
    std::cout << "\n执行测试..." << std::endl;
    MNISTSimulator test_data(200, batch_size);  // 更小的测试集
    
    float test_loss = 0.0f;
    float test_acc = 0.0f;
    size_t num_test_batches = test_data.num_batches();
    
    for (size_t batch = 0; batch < num_test_batches; ++batch) {
        auto [inputs, targets] = test_data.get_batch(batch);
        
        // 前向传播
        auto outputs = model.forward(inputs);
        auto loss = criterion.forward(outputs, targets);
        
        // 计算准确率
        float accuracy = compute_accuracy(outputs, targets);
        
        // 累计统计
        test_loss += loss.data()[0];
        test_acc += accuracy;
    }
    
    // 计算平均值
    test_loss /= num_test_batches;
    test_acc /= num_test_batches;
    
    std::cout << "测试集结果 - Loss: " << test_loss 
              << ", Accuracy: " << (test_acc * 100.0f) << "%" << std::endl;
    
    return 0;
} 
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

        std::cout << "获取批次 " << batch_idx << " - 样本数: " << actual_batch_size << std::endl;

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

        std::cout << "  创建批次张量 - x形状: [" << actual_batch_size << ", 784]"
                  << ", y形状: [" << actual_batch_size << "]" << std::endl;

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

        std::cout << "生成了 " << num_samples_ << " 个模拟MNIST样本" << std::endl;
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
        std::cout << "创建MNIST网络..." << std::endl;
        fc1_ = std::make_shared<Linear>(784, 128);  // 输入层 -> 隐藏层
        relu1_ = std::make_shared<ReLU>();
        fc2_ = std::make_shared<Linear>(128, 64);   // 隐藏层 -> 隐藏层 
        relu2_ = std::make_shared<ReLU>();
        fc3_ = std::make_shared<Linear>(64, 10);    // 隐藏层 -> 输出层
        std::cout << "网络创建完成" << std::endl;
    }

    Variable forward(const Variable& x) {
        std::cout << "MNISTNet::forward - 输入形状: [";
        auto x_shape = x.data().shape();
        for (size_t i = 0; i < x_shape.size(); ++i) {
            std::cout << x_shape[i];
            if (i < x_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // 第一层
        std::cout << "  第一层: fc1" << std::endl;
        auto h1 = fc1_->forward(x);
        auto h1_shape = h1.data().shape();
        std::cout << "  fc1输出形状: [";
        for (size_t i = 0; i < h1_shape.size(); ++i) {
            std::cout << h1_shape[i];
            if (i < h1_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  第一层: relu1" << std::endl;
        auto a1 = relu1_->forward(h1);

        // 第二层
        std::cout << "  第二层: fc2" << std::endl;
        auto h2 = fc2_->forward(a1);
        auto h2_shape = h2.data().shape();
        std::cout << "  fc2输出形状: [";
        for (size_t i = 0; i < h2_shape.size(); ++i) {
            std::cout << h2_shape[i];
            if (i < h2_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  第二层: relu2" << std::endl;
        auto a2 = relu2_->forward(h2);

        // 输出层
        std::cout << "  输出层: fc3" << std::endl;
        auto out = fc3_->forward(a2);
        auto out_shape = out.data().shape();
        std::cout << "  fc3输出形状: [";
        for (size_t i = 0; i < out_shape.size(); ++i) {
            std::cout << out_shape[i];
            if (i < out_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

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
    const size_t num_samples = 320;  // 减小样本量，简化测试
    MNISTSimulator train_data(num_samples, batch_size);
    
    // 创建模型
    MNISTNet model;
    
    // 创建损失函数和优化器
    CrossEntropyLoss criterion;
    optim::Adam optimizer(model.parameters(), 0.005f);  // 降低学习率
    
    // 训练参数
    const size_t num_epochs = 2;  // 减少epoch数，简化测试
    
    std::cout << "\n开始训练..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 训练循环
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\nEpoch " << (epoch+1) << "/" << num_epochs << std::endl;
            float total_loss = 0.0f;
            float total_acc = 0.0f;
            
            size_t num_batches = train_data.num_batches();
            std::cout << "批次总数: " << num_batches << std::endl;
            
            // 只处理少量批次进行测试
            size_t max_batches = std::min(num_batches, size_t(3));
            
            for (size_t batch = 0; batch < max_batches; ++batch) {
                std::cout << "\n处理批次 " << (batch+1) << "/" << max_batches << std::endl;
                
                // 获取一批数据
                std::cout << "获取批次数据..." << std::endl;
                auto [inputs, targets] = train_data.get_batch(batch);
                
                // 前向传播
                std::cout << "执行前向传播..." << std::endl;
                auto outputs = model.forward(inputs);
                
                std::cout << "计算损失..." << std::endl;
                // 验证输出和目标形状
                auto outputs_shape = outputs.data().shape();
                auto targets_shape = targets.data().shape();
                
                std::cout << "  输出形状: [";
                for (size_t i = 0; i < outputs_shape.size(); ++i) {
                    std::cout << outputs_shape[i];
                    if (i < outputs_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "  目标形状: [";
                for (size_t i = 0; i < targets_shape.size(); ++i) {
                    std::cout << targets_shape[i];
                    if (i < targets_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                auto loss = criterion.forward(outputs, targets);
                
                // 计算准确率
                std::cout << "计算准确率..." << std::endl;
                float accuracy = compute_accuracy(outputs, targets);
                
                // 反向传播和优化
                std::cout << "执行反向传播..." << std::endl;
                optimizer.zero_grad();
                try {
                    loss.backward();
                    
                    std::cout << "更新参数..." << std::endl;
                    optimizer.step();
                    
                    // 累计统计
                    total_loss += loss.data()[0];
                    total_acc += accuracy;
                    
                    std::cout << "批次 " << (batch+1) << " - 损失: " << loss.data()[0]
                              << ", 准确率: " << (accuracy * 100.0f) << "%" << std::endl;
                }
                catch (const std::exception& e) {
                    std::cout << "反向传播或参数更新出错: " << e.what() << std::endl;
                    std::cout << "跳过这个批次，继续训练" << std::endl;
                }
            }
            
            // 计算平均值
            if (max_batches > 0) {
                float avg_loss = total_loss / max_batches;
                float avg_acc = total_acc / max_batches;
                
                std::cout << "\nEpoch " << (epoch+1) << "/" << num_epochs << " 总结 - "
                          << "平均损失: " << avg_loss << ", "
                          << "平均准确率: " << (avg_acc * 100.0f) << "%" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        std::cout << "\n训练完成! 用时: " << duration / 1000.0f << " 秒" << std::endl;
        
        // 测试
        std::cout << "\n执行测试..." << std::endl;
        MNISTSimulator test_data(100, batch_size);  // 更小的测试集
        
        float test_loss = 0.0f;
        float test_acc = 0.0f;
        size_t num_test_batches = std::min(test_data.num_batches(), size_t(2));
        
        std::cout << "测试批次数: " << num_test_batches << std::endl;
        
        for (size_t batch = 0; batch < num_test_batches; ++batch) {
            std::cout << "\n测试批次 " << (batch+1) << "/" << num_test_batches << std::endl;
            
            // 获取数据
            auto [inputs, targets] = test_data.get_batch(batch);
            
            // 前向传播
            std::cout << "执行前向传播..." << std::endl;
            auto outputs = model.forward(inputs);
            
            // 计算损失
            std::cout << "计算损失..." << std::endl;
            auto loss = criterion.forward(outputs, targets);
            
            // 计算准确率
            std::cout << "计算准确率..." << std::endl;
            float accuracy = compute_accuracy(outputs, targets);
            
            // 累计统计
            test_loss += loss.data()[0];
            test_acc += accuracy;
            
            std::cout << "批次 " << (batch+1) << " - 损失: " << loss.data()[0]
                      << ", 准确率: " << (accuracy * 100.0f) << "%" << std::endl;
        }
        
        // 计算平均值
        if (num_test_batches > 0) {
            test_loss /= num_test_batches;
            test_acc /= num_test_batches;
            
            std::cout << "\n测试集结果 - 平均损失: " << test_loss 
                      << ", 平均准确率: " << (test_acc * 100.0f) << "%" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "\n运行过程中出错: " << e.what() << std::endl;
    }
    
    return 0;
} 
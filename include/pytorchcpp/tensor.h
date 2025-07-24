#pragma once

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <functional>
#include <iostream>

namespace pytorchcpp {

/**
 * @brief 张量类，实现多维数组的表示和操作
 */
class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad = false);
    
    // 拷贝和移动构造/赋值
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 析构函数
    ~Tensor();
    
    // 工厂函数
    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor randn(const std::vector<size_t>& shape, bool requires_grad = false);
    
    // 基本属性
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t numel() const;
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    
    // 索引和访问
    float& at(const std::vector<size_t>& indices);
    float at(const std::vector<size_t>& indices) const;
    float& operator[](size_t index);
    float operator[](size_t index) const;
    
    // 基本运算
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const;
    
    // 运算符重载
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // 归约操作
    Tensor sum(int64_t dim = -1, bool keepdim = false) const;
    Tensor mean(int64_t dim = -1, bool keepdim = false) const;
    
    // 形状操作
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor view(const std::vector<size_t>& new_shape) const;
    
    // 打印和序列化
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    std::string to_string() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;  // PIMPL模式
};

} // namespace pytorchcpp 
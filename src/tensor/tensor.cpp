#include <pytorchcpp/tensor.h>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace pytorchcpp {

// PIMPL实现
struct Tensor::Impl {
    std::vector<size_t> shape;
    std::vector<float> data;
    bool requires_grad;
    
    Impl() : requires_grad(false) {}
    
    Impl(const std::vector<size_t>& shape, bool requires_grad) 
        : shape(shape), requires_grad(requires_grad) {
        size_t size = std::accumulate(shape.begin(), shape.end(), 
                                     static_cast<size_t>(1), std::multiplies<>());
        data.resize(size, 0.0f);
    }
    
    Impl(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad) 
        : shape(shape), data(data), requires_grad(requires_grad) {
        size_t expected_size = std::accumulate(shape.begin(), shape.end(), 
                                              static_cast<size_t>(1), std::multiplies<>());
        if (data.size() != expected_size) {
            throw std::invalid_argument("Data size does not match shape dimensions");
        }
    }
    
    size_t index_to_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Indices dimension mismatch");
        }
        
        size_t offset = 0;
        size_t multiplier = 1;
        
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        
        return offset;
    }
};

// 构造函数
Tensor::Tensor() : pImpl(std::make_unique<Impl>()) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : pImpl(std::make_unique<Impl>(shape, requires_grad)) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad)
    : pImpl(std::make_unique<Impl>(shape, data, requires_grad)) {}

// 拷贝和移动构造/赋值
Tensor::Tensor(const Tensor& other) : pImpl(std::make_unique<Impl>(*other.pImpl)) {}

Tensor::Tensor(Tensor&& other) noexcept = default;

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        *pImpl = *other.pImpl;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept = default;

// 析构函数
Tensor::~Tensor() = default;

// 工厂函数
Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 
                                 static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> data(size, 1.0f);
    return Tensor(shape, data, requires_grad);
}

Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 
                                 static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> data(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : data) {
        val = dist(gen);
    }
    
    return Tensor(shape, data, requires_grad);
}

// 基本属性
std::vector<size_t> Tensor::shape() const {
    return pImpl->shape;
}

size_t Tensor::ndim() const {
    return pImpl->shape.size();
}

size_t Tensor::numel() const {
    return pImpl->data.size();
}

bool Tensor::requires_grad() const {
    return pImpl->requires_grad;
}

void Tensor::set_requires_grad(bool requires_grad) {
    pImpl->requires_grad = requires_grad;
}

// 索引和访问
float& Tensor::at(const std::vector<size_t>& indices) {
    size_t offset = pImpl->index_to_offset(indices);
    return pImpl->data[offset];
}

float Tensor::at(const std::vector<size_t>& indices) const {
    size_t offset = pImpl->index_to_offset(indices);
    return pImpl->data[offset];
}

float& Tensor::operator[](size_t index) {
    if (index >= pImpl->data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return pImpl->data[index];
}

float Tensor::operator[](size_t index) const {
    if (index >= pImpl->data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return pImpl->data[index];
}

// 基本运算
Tensor Tensor::add(const Tensor& other) const {
    if (pImpl->shape != other.pImpl->shape) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    std::vector<float> result_data(pImpl->data.size());
    for (size_t i = 0; i < pImpl->data.size(); ++i) {
        result_data[i] = pImpl->data[i] + other.pImpl->data[i];
    }
    
    return Tensor(pImpl->shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}

Tensor Tensor::sub(const Tensor& other) const {
    if (pImpl->shape != other.pImpl->shape) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    std::vector<float> result_data(pImpl->data.size());
    for (size_t i = 0; i < pImpl->data.size(); ++i) {
        result_data[i] = pImpl->data[i] - other.pImpl->data[i];
    }
    
    return Tensor(pImpl->shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (pImpl->shape != other.pImpl->shape) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    
    std::vector<float> result_data(pImpl->data.size());
    for (size_t i = 0; i < pImpl->data.size(); ++i) {
        result_data[i] = pImpl->data[i] * other.pImpl->data[i];
    }
    
    return Tensor(pImpl->shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}

Tensor Tensor::div(const Tensor& other) const {
    if (pImpl->shape != other.pImpl->shape) {
        throw std::invalid_argument("Tensor shapes must match for element-wise division");
    }
    
    std::vector<float> result_data(pImpl->data.size());
    for (size_t i = 0; i < pImpl->data.size(); ++i) {
        if (other.pImpl->data[i] == 0) {
            throw std::invalid_argument("Division by zero");
        }
        result_data[i] = pImpl->data[i] / other.pImpl->data[i];
    }
    
    return Tensor(pImpl->shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 简单实现2D矩阵乘法
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    size_t m = pImpl->shape[0];
    size_t k = pImpl->shape[1];
    
    if (k != other.pImpl->shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    
    size_t n = other.pImpl->shape[1];
    std::vector<size_t> result_shape = {m, n};
    std::vector<float> result_data(m * n, 0.0f);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                result_data[i * n + j] += pImpl->data[i * k + k_idx] * other.pImpl->data[k_idx * n + j];
            }
        }
    }
    
    return Tensor(result_shape, result_data, pImpl->requires_grad || other.pImpl->requires_grad);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim()) {
        throw std::invalid_argument("Dimension out of range");
    }
    
    std::vector<size_t> new_shape = pImpl->shape;
    std::swap(new_shape[dim0], new_shape[dim1]);
    
    std::vector<float> result_data(pImpl->data.size());
    
    // 这里只实现2D转置作为示例
    if (ndim() == 2 && dim0 == 0 && dim1 == 1) {
        size_t rows = pImpl->shape[0];
        size_t cols = pImpl->shape[1];
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result_data[j * rows + i] = pImpl->data[i * cols + j];
            }
        }
    } else {
        throw std::runtime_error("General transpose not implemented yet");
    }
    
    return Tensor(new_shape, result_data, pImpl->requires_grad);
}

// 运算符重载
Tensor Tensor::operator+(const Tensor& other) const {
    return add(other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return sub(other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return mul(other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return div(other);
}

// 归约操作
Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    if (dim == -1) {
        // 全局求和
        float sum_val = std::accumulate(pImpl->data.begin(), pImpl->data.end(), 0.0f);
        return Tensor({1}, {sum_val}, pImpl->requires_grad);
    }
    
    throw std::runtime_error("Dimension-specific reduction not implemented yet");
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    if (dim == -1) {
        // 全局平均
        float sum_val = std::accumulate(pImpl->data.begin(), pImpl->data.end(), 0.0f);
        float mean_val = sum_val / pImpl->data.size();
        return Tensor({1}, {mean_val}, pImpl->requires_grad);
    }
    
    throw std::runtime_error("Dimension-specific reduction not implemented yet");
}

// 形状操作
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                     static_cast<size_t>(1), std::multiplies<>());
    if (new_size != pImpl->data.size()) {
        throw std::invalid_argument("New shape must have the same number of elements");
    }
    
    return Tensor(new_shape, pImpl->data, pImpl->requires_grad);
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    // 在这个简单实现中，view和reshape行为相同
    return reshape(new_shape);
}

// 打印和序列化
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

std::string Tensor::to_string() const {
    if (pImpl->data.empty()) {
        return "Tensor()";
    }
    
    std::ostringstream oss;
    oss << "Tensor(";
    
    if (ndim() <= 1) {
        // 0D或1D张量
        oss << "[";
        for (size_t i = 0; i < pImpl->data.size(); ++i) {
            oss << pImpl->data[i];
            if (i < pImpl->data.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
    } else if (ndim() == 2) {
        // 2D张量
        size_t rows = pImpl->shape[0];
        size_t cols = pImpl->shape[1];
        
        oss << "[";
        for (size_t i = 0; i < rows; ++i) {
            oss << "[";
            for (size_t j = 0; j < cols; ++j) {
                oss << pImpl->data[i * cols + j];
                if (j < cols - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
            if (i < rows - 1) {
                oss << ", ";
            }
        }
        oss << "]";
    } else {
        // 高维张量
        oss << "shape=" << "[";
        for (size_t i = 0; i < pImpl->shape.size(); ++i) {
            oss << pImpl->shape[i];
            if (i < pImpl->shape.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
    }
    
    if (pImpl->requires_grad) {
        oss << ", requires_grad=True";
    }
    
    oss << ")";
    return oss.str();
}

} // namespace pytorchcpp 
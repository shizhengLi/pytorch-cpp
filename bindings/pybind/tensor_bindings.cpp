#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pytorchcpp/tensor.h>

namespace py = pybind11;
using namespace pytorchcpp;

void init_tensor(py::module_ &m) {
    py::class_<Tensor>(m, "Tensor")
        // 构造函数
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&, bool>(),
             py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)
        
        // 工厂方法
        .def_static("zeros", &Tensor::zeros,
                   py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("ones", &Tensor::ones,
                   py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("randn", &Tensor::randn,
                   py::arg("shape"), py::arg("requires_grad") = false)
        
        // 属性
        .def("shape", &Tensor::shape)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)
        .def("requires_grad", &Tensor::requires_grad)
        .def("set_requires_grad", &Tensor::set_requires_grad)
        
        // 索引和访问
        .def("at", static_cast<float& (Tensor::*)(const std::vector<size_t>&)>(&Tensor::at),
             py::arg("indices"), py::return_value_policy::reference)
        .def("at", static_cast<float (Tensor::*)(const std::vector<size_t>&) const>(&Tensor::at),
             py::arg("indices"))
        .def("__getitem__", static_cast<float (Tensor::*)(size_t) const>(&Tensor::operator[]),
             py::arg("index"))
        .def("__setitem__", [](Tensor &t, size_t index, float value) {
            t[index] = value;
        })
        
        // 基本运算
        .def("add", &Tensor::add, py::arg("other"))
        .def("sub", &Tensor::sub, py::arg("other"))
        .def("mul", &Tensor::mul, py::arg("other"))
        .def("div", &Tensor::div, py::arg("other"))
        .def("matmul", &Tensor::matmul, py::arg("other"))
        .def("transpose", &Tensor::transpose, py::arg("dim0") = 0, py::arg("dim1") = 1)
        
        // 运算符重载
        .def("__add__", &Tensor::operator+, py::arg("other"))
        .def("__sub__", &Tensor::operator-, py::arg("other"))
        .def("__mul__", &Tensor::operator*, py::arg("other"))
        .def("__truediv__", &Tensor::operator/, py::arg("other"))
        
        // 归约操作
        .def("sum", &Tensor::sum, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("mean", &Tensor::mean, py::arg("dim") = -1, py::arg("keepdim") = false)
        
        // 形状操作
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("view", &Tensor::view, py::arg("new_shape"))
        
        // 字符串表示
        .def("__repr__", &Tensor::to_string)
        .def("__str__", &Tensor::to_string);
        
    // 从NumPy数组创建Tensor的转换函数
    m.def("from_numpy", [](py::array_t<float> array) {
        py::buffer_info info = array.request();
        std::vector<size_t> shape;
        for (auto dim : info.shape) {
            shape.push_back(static_cast<size_t>(dim));
        }
        std::vector<float> data(static_cast<float*>(info.ptr), 
                              static_cast<float*>(info.ptr) + info.size);
        return Tensor(shape, data);
    });
    
    // 将Tensor转换为NumPy数组
    m.def("to_numpy", [](const Tensor &tensor) {
        std::vector<size_t> shape = tensor.shape();
        std::vector<ssize_t> numpy_shape;
        for (auto dim : shape) {
            numpy_shape.push_back(static_cast<ssize_t>(dim));
        }
        
        size_t size = tensor.numel();
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = tensor[i];
        }
        
        return py::array_t<float>(numpy_shape, data.data());
    });
} 
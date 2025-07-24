#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void init_tensor(py::module_ &);
void init_autograd(py::module_ &);
void init_nn(py::module_ &);
void init_optim(py::module_ &);

PYBIND11_MODULE(pytorchcpp, m) {
    m.doc() = "PyTorchCPP: A lightweight deep learning library in C++";
    
    // 创建子模块
    auto tensor_submodule = m.def_submodule("tensor", "Tensor operations");
    auto autograd_submodule = m.def_submodule("autograd", "Automatic differentiation");
    auto nn_submodule = m.def_submodule("nn", "Neural network modules");
    auto optim_submodule = m.def_submodule("optim", "Optimization algorithms");
    
    // 初始化各个子模块
    init_tensor(m);         // 在主模块中添加Tensor类
    init_autograd(m);       // 在主模块中添加Variable类
    init_nn(nn_submodule);  // 添加神经网络模块
    init_optim(optim_submodule); // 添加优化器模块
} 
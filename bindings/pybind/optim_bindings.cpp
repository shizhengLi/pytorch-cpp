#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pytorchcpp/optim.h>

namespace py = pybind11;
using namespace pytorchcpp;
using namespace pytorchcpp::optim;

void init_optim(py::module_ &m) {
    // 优化器基类
    py::class_<Optimizer>(m, "Optimizer")
        .def("zero_grad", &Optimizer::zero_grad)
        .def("set_learning_rate", &Optimizer::set_learning_rate)
        .def("get_learning_rate", &Optimizer::get_learning_rate);
        
    // SGD优化器
    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init<const std::unordered_map<std::string, Variable>&, float, float, float>(),
             py::arg("parameters"),
             py::arg("learning_rate") = 0.01f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f)
        .def("step", &SGD::step);
        
    // Adam优化器
    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init<const std::unordered_map<std::string, Variable>&, float, float, float, float, float>(),
             py::arg("parameters"),
             py::arg("learning_rate") = 0.001f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f,
             py::arg("weight_decay") = 0.0f)
        .def("step", &Adam::step);
} 
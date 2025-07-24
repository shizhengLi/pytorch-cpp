#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pytorchcpp/nn.h>

namespace py = pybind11;
using namespace pytorchcpp;
using namespace pytorchcpp::nn;

void init_nn(py::module_ &m) {
    // 模块基类
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("__call__", &Module::operator())
        .def("parameters", &Module::parameters);
        
    // 线性层
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &Linear::forward);
        
    // 激活函数
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);
        
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward);
        
    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward);
        
    // Sequential容器
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add)
        .def("forward", &Sequential::forward)
        .def("parameters", &Sequential::parameters);
        
    // 损失函数
    py::class_<MSELoss, Module, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", static_cast<Variable (MSELoss::*)(const Variable&, const Variable&)>(&MSELoss::forward))
        .def("__call__", [](MSELoss& self, const Variable& input, const Variable& target) {
            return self.forward(input, target);
        });
        
    py::class_<CrossEntropyLoss, Module, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", static_cast<Variable (CrossEntropyLoss::*)(const Variable&, const Variable&)>(&CrossEntropyLoss::forward))
        .def("__call__", [](CrossEntropyLoss& self, const Variable& input, const Variable& target) {
            return self.forward(input, target);
        });
} 
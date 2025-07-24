#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pytorchcpp/autograd.h>
#include <sstream>

namespace py = pybind11;
using namespace pytorchcpp;

void init_autograd(py::module_ &m) {
    py::class_<Variable>(m, "Variable")
        // 构造函数
        .def(py::init<>())
        .def(py::init<const Tensor&, bool>(),
             py::arg("data"), py::arg("requires_grad") = false)
        
        // 拷贝和移动
        .def(py::init<const Variable&>())
        
        // 访问器
        .def("data", &Variable::data)
        .def("grad", &Variable::grad)
        .def("requires_grad", &Variable::requires_grad)
        .def("set_requires_grad", &Variable::set_requires_grad)
        
        // 梯度操作
        .def("zero_grad", &Variable::zero_grad)
        .def("backward", &Variable::backward, py::arg("grad") = Tensor())
        
        // 运算操作
        .def("add", &Variable::add)
        .def("sub", &Variable::sub)
        .def("mul", &Variable::mul)
        .def("div", &Variable::div)
        .def("matmul", &Variable::matmul)
        
        // 运算符重载
        .def("__add__", &Variable::operator+)
        .def("__sub__", &Variable::operator-)
        .def("__mul__", &Variable::operator*)
        .def("__truediv__", &Variable::operator/)
        
        // 字符串表示
        .def("__repr__", [](const Variable& var) {
            std::stringstream ss;
            ss << "Variable containing:" << std::endl;
            ss << var.data().to_string();
            if (var.requires_grad()) {
                ss << std::endl << "requires_grad=True";
            }
            return ss.str();
        })
        .def("__str__", [](const Variable& var) {
            std::stringstream ss;
            ss << "Variable containing:" << std::endl;
            ss << var.data().to_string();
            if (var.requires_grad()) {
                ss << std::endl << "requires_grad=True";
            }
            return ss.str();
        });
} 
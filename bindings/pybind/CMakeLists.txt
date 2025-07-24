pybind11_add_module(pytorchcpp_python
    bindings.cpp
    tensor_bindings.cpp
    autograd_bindings.cpp
    nn_bindings.cpp
    optim_bindings.cpp
)

target_link_libraries(pytorchcpp_python PRIVATE
    tensor
    autograd
    nn
    optim
)

# 如果使用setuptools安装，需要将Python模块名设置为目标名称
set_target_properties(pytorchcpp_python PROPERTIES OUTPUT_NAME "pytorchcpp")

# 指定安装位置
install(TARGETS pytorchcpp_python DESTINATION .) 